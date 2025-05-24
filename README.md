# Chungoid

**Transform ideas into production-ready software with one command.**

Chungoid is an autonomous AI development toolkit that converts natural language goals into complete, working projects. Using coordinated AI agents and intelligent automation, it handles everything from architecture decisions to dependency management, letting you focus on what you want to build rather than how to build it.

```bash
# Create a goal file
echo "Build a Python FastAPI REST API with user authentication" > goal.txt

# Generate complete project
chungoid build --goal-file goal.txt --project-dir ./my-api

# Result: A production-ready FastAPI application with auth, tests, and documentation
```

## Key Features

- **One-Command Project Generation**: From goal to working code in minutes
- **Autonomous Development**: AI agents handle architecture, coding, testing, and deployment
- **Intelligent Dependency Management**: Automatically resolves and installs requirements
- **Adaptive Learning**: Learns from past executions to improve future builds
- **Multi-Language Support**: Python, JavaScript, TypeScript, and more
- **State Persistence**: Resume interrupted builds, track project evolution
- **Production-Ready**: Generates deployment configs, tests, and documentation

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git (for project initialization)

### Installation

Currently available as development installation:

```bash
# Clone the repository
git clone https://github.com/your-org/chungoid-core.git
cd chungoid-core

# Install in development mode
pip install -e .

# Verify installation
chungoid --help
```

### Your First Project

1. **Create a goal file**:
```bash
echo "Create a Python CLI tool using Click with hello and goodbye commands" > my-goal.txt
```

2. **Initialize project directory** (optional):
```bash
chungoid init my-cli-tool
cd my-cli-tool
```

3. **Build your project**:
```bash
chungoid build --goal-file ../my-goal.txt --project-dir .
```

4. **Check the results**:
```bash
# View project status
chungoid status

# Your working CLI tool is ready!
python -m your_cli hello
```

## Writing Effective Goals

The quality of your goal description directly impacts the generated project. Here are examples of effective goals:

### Good Goals

```text
Create a Python FastAPI REST API with:
- User registration and JWT authentication
- CRUD operations for a blog post model
- SQLAlchemy with PostgreSQL
- Automatic API documentation
- Docker containerization
- Pytest test suite
```

```text
Build a React TypeScript web app that:
- Displays a dashboard with charts using Chart.js
- Fetches data from a REST API
- Has responsive design with Tailwind CSS
- Includes user login with session management
- Uses Vite for build tooling
```

### Avoid Vague Goals

```text
Make a website  # Too vague
Build something cool  # No specific requirements
Create an app  # Missing technology and functionality details
```

### Goal Writing Tips

- **Be specific about technology stack**: "Python Flask" vs "Python"
- **Define core features clearly**: List the main functionality needed
- **Include deployment preferences**: Docker, cloud platform, etc.
- **Specify testing requirements**: Unit tests, integration tests
- **Mention UI/UX preferences**: Responsive, specific frameworks

## Core Commands

### `chungoid build` - Primary Command

Build a complete project from a goal description.

```bash
chungoid build --goal-file GOAL_FILE [OPTIONS]
```

**Required:**
- `--goal-file FILE`: Path to your goal description file

**Options:**
- `--project-dir DIR`: Target directory (default: current directory)
- `--initial-context JSON`: Additional context as JSON string
- `--tags TAGS`: Comma-separated tags for organization
- `--run-id ID`: Custom run identifier

**Examples:**

```bash
# Basic usage
chungoid build --goal-file goal.txt

# Specify target directory
chungoid build --goal-file goal.txt --project-dir ./new-project

# Add context for better results
chungoid build --goal-file goal.txt --initial-context '{"framework": "fastapi", "database": "postgresql"}'

# Tag for organization
chungoid build --goal-file goal.txt --tags "api,microservice,production"
```

### `chungoid init` - Project Setup

Initialize a new Chungoid project structure.

```bash
chungoid init PROJECT_DIR
```

Creates a `.chungoid/` directory with configuration files and project tracking.

### `chungoid status` - Project Inspection

Check the status of a Chungoid project.

```bash
chungoid status [PROJECT_DIR] [--json]
```

Shows build progress, run history, and project metadata.

### Advanced Commands

For complex workflows and automation:

- `chungoid flow run`: Execute custom workflow definitions
- `chungoid flow resume`: Resume interrupted executions
- `chungoid metrics`: View execution analytics
- `chungoid utils show-config`: Display current configuration

## Configuration

### LLM Provider Setup

Chungoid uses **LiteLLM** to support a wide variety of LLM providers with a unified interface. This means you can use OpenAI, Anthropic, Ollama, Azure, Google, and many other providers.

#### Quick Setup

**OpenAI (recommended for cloud):**
```bash
export OPENAI_API_KEY="sk-your-openai-api-key"
```

**Anthropic (Claude):**
```bash
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

**Ollama (local models):**
```bash
# Start Ollama server first
ollama serve

# No API key needed for local Ollama
```

#### Configuration File

Create `.chungoid/project_config.yaml` in your project or update global `config.yaml`:

```yaml
llm_manager:
  provider_type: "litellm"
  # OpenAI models
  default_model: "gpt-4-turbo-preview"
  # Or Anthropic models  
  # default_model: "claude-3-opus-20240229"
  # Or local Ollama models
  # default_model: "ollama/mistral"
  # base_url: "http://localhost:11434"  # For Ollama
```

#### Supported Providers & Models

| Provider | Model Examples | Setup |
|----------|-----------------|--------|
| **OpenAI** | `gpt-4-turbo-preview`, `gpt-3.5-turbo` | Set `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-opus-20240229`, `claude-3-sonnet-20240229` | Set `ANTHROPIC_API_KEY` |
| **Ollama** | `ollama/mistral`, `ollama/llama2`, `ollama/codellama` | Local server, no API key |
| **Azure OpenAI** | `azure/your-deployment-name` | Set `AZURE_API_KEY`, `AZURE_API_BASE` |
| **Google** | `gemini-pro`, `gemini-pro-vision` | Set `GOOGLE_API_KEY` |
| **HuggingFace** | `huggingface/model-name` | Set `HF_TOKEN` (for private models) |

> **📖 Detailed Setup Guide**: See `docs/guides/litellm_setup.md` for comprehensive configuration instructions for all providers.

### Environment Variables (Legacy)

### Project Configuration

Each project can have custom settings in `.chungoid/chungoid_config.yaml`:

```yaml
# .chungoid/chungoid_config.yaml
llm:
  provider: openai
  model: gpt-4-turbo
  api_key: ${CHUNGOID_LLM_API_KEY}

orchestrator:
  max_retries: 3
  failure_recovery: true

agents:
  enable_learning: true
  fallback_strategy: "graceful_degradation"
```

## How It Works

Chungoid uses a sophisticated autonomous development pipeline:

```
Goal File → Master Planner → Execution Plan → Specialized Agents → Working Project
```

### 1. **Goal Analysis**
The Master Planner Agent analyzes your goal and creates a detailed execution plan

### 2. **Agent Orchestration** 
Specialized agents execute the plan:

**Planning & Coordination Agents:**
- **MasterPlannerAgent**: Creates detailed execution plans from user goals
- **MasterPlannerReviewerAgent**: Reviews and adjusts plans when issues arise
- **ArchitectAgent**: Makes system architecture and design decisions

**Development Agents:**
- **EnvironmentBootstrapAgent**: Sets up project structure and multi-language environments
- **DependencyManagementAgent_v1**: Intelligent dependency resolution and installation
- **CodeGeneratorAgent**: Writes application code following best practices
- **TestGeneratorAgent**: Creates comprehensive test suites
- **SystemFileSystemAgent**: Handles file operations and project structure

**Quality Assurance Agents:**
- **SystemTestRunnerAgent**: Executes tests and manages test workflows
- **TestFailureAnalysisAgent_v1**: Sophisticated analysis and automated fixing of test failures

**Knowledge Management:**
- **ProjectChromaManagerAgent**: Manages knowledge storage and retrieval using ChromaDB

### 3. **Tool Integration**
Agents use 45+ specialized tools:
- **Filesystem Suite**: Smart file operations and project scanning
- **Terminal Suite**: Safe command execution and dependency management
- **ChromaDB Suite**: Learning and context management
- **Content Suite**: Documentation and template processing

### 4. **Continuous Learning**
ChromaDB integration enables the system to learn from each build, improving future projects.

## Examples

### Python FastAPI Microservice

**Goal**: `Create a FastAPI microservice with PostgreSQL, Docker, and comprehensive testing`

**Generated Project Structure**:
```
my-api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # SQLAlchemy models
│   ├── database.py          # Database configuration
│   └── routes/              # API endpoints
├── tests/                   # Pytest test suite
├── docker-compose.yml       # Development environment
├── Dockerfile               # Production container
├── requirements.txt         # Dependencies
└── README.md               # Project documentation
```

### React TypeScript Dashboard

**Goal**: `Build a React TypeScript dashboard with charts, responsive design, and API integration`

**Generated Features**:
- Vite build configuration
- TypeScript setup with strict mode
- Chart.js integration for data visualization
- Tailwind CSS for responsive design
- Axios for API communication
- Jest and Testing Library for testing

## Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# If installation fails, try upgrading pip
pip install --upgrade pip
pip install -e .
```

**Build Failures:**
```bash
# Check project status for detailed error information
chungoid status --json

# View logs
cat .chungoid/chungoid_status.json
```

**LLM Provider Issues:**
```bash
# Verify your API key is set
echo $CHUNGOID_LLM_API_KEY

# Test with mock provider (development only)
chungoid build --goal-file goal.txt --use-mock-llm-provider
```

### Getting Help

- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions for usage questions
- **Documentation**: Check `/docs` directory for detailed guides

## Optional: MCP Server Integration

Chungoid can optionally run as an MCP (Model Context Protocol) server for integration with AI development environments:

```bash
# Start MCP server (advanced usage)
chungoid-server

# Configure in your AI tool
{
  "mcpServers": {
    "chungoid": {
      "command": "chungoid-server",
      "args": []
    }
  }
}
```

> **Note**: Most users should focus on the CLI commands above. MCP integration is for advanced workflows and AI tool integrations.

## Contributing

We welcome contributions! Here's how to get started:

### Development Setup

```bash
# Clone and install for development
git clone https://github.com/your-org/chungoid-core.git
cd chungoid-core
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
flake8 src/ tests/
```

### Areas for Contribution

- **New Agents**: Specialized agents for different domains
- **MCP Tools**: Additional tool integrations
- **Language Support**: Support for new programming languages
- **Templates**: Project templates for common use cases
- **Documentation**: Examples, guides, and tutorials

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all functions
- Include docstrings for public APIs
- Write tests for new functionality
- Update documentation for user-facing changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [ChromaDB](https://www.trychroma.com/) - AI-native vector database
- [FastMCP](https://github.com/jlowin/fastmcp) - MCP protocol implementation
- [Rich](https://rich.readthedocs.io/) - Beautiful terminal output
- [Pydantic](https://pydantic-docs.helpmanual.io/) - Data validation and settings

---

**Ready to transform your ideas into code? [Get started](#quick-start) with your first Chungoid project!**