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

Chungoid uses a **modern hierarchical configuration system** with automatic environment variable integration and validation. The system supports both global and project-specific configurations with intelligent merging.

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

#### Configuration Files

**Project-specific configuration** (`.chungoid/config.yaml`):
```yaml
# .chungoid/config.yaml
llm:
  provider: "litellm"
  default_model: "gpt-4o-mini-2024-07-18"  # Cost-effective option
  # Or use other models:
  # default_model: "claude-3-sonnet-20240229"  # Anthropic
  # default_model: "ollama/mistral"             # Local Ollama
  max_tokens_per_request: 8000
  temperature: 0.1

orchestrator:
  max_retries: 3
  failure_recovery: true

agents:
  enable_learning: true
  fallback_strategy: "graceful_degradation"

chromadb:
  host: "localhost"
  port: 8000
  # For persistent mode, these settings are automatically configured
  default_collection_prefix: "chungoid"
```

**Global configuration** (`~/.chungoid/config.yaml`):
```yaml
# Global defaults - applies to all projects
llm:
  provider: "litellm" 
  default_model: "gpt-4o-mini-2024-07-18"
  max_tokens_per_request: 8000

logging:
  level: "INFO"
  enable_structured_logging: true  # Use JSON logging
  enable_file_logging: true
  log_directory: "logs"
  
chromadb:
  host: "localhost"
  port: 8000
  default_collection_prefix: "chungoid"
```

#### Environment Variables

The configuration system automatically reads environment variables with the `CHUNGOID_` prefix:

```bash
# LLM Configuration
export CHUNGOID_LLM_PROVIDER="litellm"
export CHUNGOID_LLM_DEFAULT_MODEL="gpt-4o-mini-2024-07-18"

# For specific providers, use standard variables:
export OPENAI_API_KEY="sk-your-key"              # OpenAI
export ANTHROPIC_API_KEY="sk-ant-your-key"       # Anthropic  
export OLLAMA_BASE_URL="http://localhost:11434"  # Ollama

# Logging
export CHUNGOID_LOG_LEVEL="INFO"
export CHUNGOID_ENABLE_DEBUG="false"

# ChromaDB
export CHUNGOID_CHROMADB_HOST="localhost"
export CHUNGOID_CHROMADB_PORT="8000"
```

#### Configuration Priority Order

The system merges configuration from multiple sources (highest priority first):

1. **Environment variables** (`CHUNGOID_*`)
2. **Project configuration** (`.chungoid/config.yaml`)
3. **Global configuration** (`~/.chungoid/config.yaml`)
4. **Built-in defaults**

#### Supported Providers & Models

| Provider | Model Examples | Setup |
|----------|-----------------|--------|
| **OpenAI** | `gpt-4o-mini-2024-07-18`, `gpt-4-turbo-preview` | Set `OPENAI_API_KEY` |
| **Anthropic** | `claude-3-opus-20240229`, `claude-3-sonnet-20240229` | Set `ANTHROPIC_API_KEY` |
| **Ollama** | `ollama/mistral`, `ollama/llama2`, `ollama/codellama` | Local server, no API key |
| **Azure OpenAI** | `azure/your-deployment-name` | Set `AZURE_API_KEY`, `AZURE_API_BASE` |
| **Google** | `gemini-pro`, `gemini-pro-vision` | Set `GOOGLE_API_KEY` |
| **HuggingFace** | `huggingface/model-name` | Set `HF_TOKEN` (for private models) |

> **📖 Detailed Setup Guide**: See `docs/guides/litellm_setup.md` for comprehensive configuration instructions for all providers.

#### Viewing Current Configuration

Check your current configuration settings:

```bash
# Show effective configuration (merged from all sources)
chungoid utils show-config

# Show raw project configuration file
chungoid utils show-config --raw

# Show configuration for specific project
chungoid utils show-config --project-dir /path/to/project
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

# View logs in project status
cat .chungoid/chungoid_status.json

# Check configuration
chungoid utils show-config
```

**Configuration Issues:**
```bash
# Verify your configuration is loaded correctly
chungoid utils show-config

# Check environment variables
env | grep CHUNGOID

# Test with specific model
export CHUNGOID_LLM_DEFAULT_MODEL="gpt-4o-mini-2024-07-18"
chungoid build --goal-file goal.txt
```

**LLM Provider Issues:**
```bash
# Verify your API key is set
echo $OPENAI_API_KEY
# or
echo $ANTHROPIC_API_KEY

# Check provider configuration
chungoid utils show-config | grep -A 10 "llm"

# Test with different model
export CHUNGOID_LLM_DEFAULT_MODEL="gpt-3.5-turbo"
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