# User Guide

This comprehensive guide covers all aspects of using Chungoid for autonomous software development.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Command Line Interface](#command-line-interface)
3. [Project Configuration](#project-configuration)
4. [Writing Effective Goals](#writing-effective-goals)
5. [Agent Ecosystem](#agent-ecosystem)
6. [MCP Tools](#mcp-tools)
7. [Workflows and Execution](#workflows-and-execution)
8. [Human-in-the-Loop](#human-in-the-loop)
9. [Monitoring and Debugging](#monitoring-and-debugging)
10. [Best Practices](#best-practices)

## Core Concepts

### Autonomous Execution

Chungoid operates on the principle of **autonomous execution** - agents work independently using protocols and tools until tasks are complete. Unlike traditional AI coding assistants that require constant guidance, Chungoid agents:

- Iterate autonomously on tasks
- Validate their work against success criteria
- Self-correct based on feedback
- Continue until completion or human intervention

### Agent-Based Architecture

The system uses specialized agents for different aspects of software development:

- **Goal Analysis Agents** - Understand and refine user requirements
- **Architecture Agents** - Design system blueprints and execution plans
- **Environment Agents** - Set up development environments
- **Code Generation Agents** - Create production-ready code
- **Testing Agents** - Generate and execute comprehensive tests
- **Documentation Agents** - Produce project documentation

### MCP Tool Ecosystem

The Model Context Protocol (MCP) enables agents to use over 45 specialized tools:

- **File System Tools** - Project scanning, file operations
- **ChromaDB Tools** - Vector storage and semantic search
- **Analysis Tools** - Code analysis and dependency management
- **Testing Tools** - Test execution and validation

## Command Line Interface

### Basic Commands

#### `chungoid init`
Initialize a new Chungoid project:

```bash
chungoid init [--project-dir DIR] [--template TEMPLATE]
```

Options:
- `--project-dir`: Target directory (default: current directory)
- `--template`: Project template to use

#### `chungoid build`
Build a complete project from a goal:

```bash
chungoid build --goal-file GOAL_FILE --project-dir PROJECT_DIR [OPTIONS]
```

Required:
- `--goal-file`: Path to goal description file
- `--project-dir`: Target project directory

Options:
- `--tags`: Comma-separated tags for categorization
- `--pause-after`: Pause after specific stage (blueprint, plan, code)
- `--debug`: Enable debug logging
- `--config`: Custom configuration file path

#### `chungoid flow run`
Execute a predefined workflow:

```bash
chungoid flow run --flow-yaml FLOW_FILE --project-dir PROJECT_DIR [OPTIONS]
```

Required:
- `--flow-yaml`: Path to workflow definition file
- `--project-dir`: Target project directory

#### `chungoid resume`
Resume paused execution:

```bash
chungoid resume --project-dir PROJECT_DIR
```

#### `chungoid status`
Check project status:

```bash
chungoid status --project-dir PROJECT_DIR
```

### Advanced Commands

#### `chungoid interactive`
Interactive requirements gathering:

```bash
chungoid interactive --goal "Build a web application"
```

#### `chungoid validate`
Validate project artifacts:

```bash
chungoid validate --project-dir PROJECT_DIR [--artifact-type TYPE]
```

#### `chungoid export`
Export project artifacts:

```bash
chungoid export --project-dir PROJECT_DIR --format [json|yaml|html]
```

## Project Configuration

### Configuration File Structure

The `.chungoid/config.yaml` file controls all aspects of Chungoid behavior:

```yaml
# LLM Provider Configuration
llm:
  provider: "openai"                    # openai, anthropic, ollama, azure
  default_model: "gpt-4o-mini-2024-07-18"
  api_key: "${OPENAI_API_KEY}"
  api_base_url: null                    # For custom endpoints
  temperature: 0.7
  max_tokens: 4000
  timeout: 300

# ChromaDB Configuration
chroma:
  persist_directory: ".chungoid/chroma_db"
  collection_prefix: "project"
  embedding_model: "all-MiniLM-L6-v2"
  max_batch_size: 100

# Agent Configuration
agents:
  timeout: 300                          # Agent timeout in seconds
  max_retries: 3                        # Maximum retry attempts
  retry_delay: 5                        # Delay between retries
  parallel_execution: true              # Enable parallel agent execution
  
  # Agent-specific settings
  ProductAnalystAgent_v1:
    confidence_threshold: 0.8
    max_requirements: 50
    
  CodeGeneratorAgent_v1:
    max_file_size: 10000
    include_comments: true
    style_guide: "pep8"

# MCP Tools Configuration
mcp_tools:
  filesystem:
    max_file_size: 1048576              # 1MB limit
    excluded_extensions: [".pyc", ".log"]
    
  chroma:
    max_documents_per_query: 20
    similarity_threshold: 0.7
    
  analysis:
    include_dependencies: true
    security_scanning: true

# Execution Configuration
execution:
  max_parallel_stages: 4
  checkpoint_frequency: 5               # Save state every N stages
  auto_recovery: true
  human_intervention_points: ["blueprint", "code"]

# Logging Configuration
logging:
  level: "INFO"                         # DEBUG, INFO, WARNING, ERROR
  file: ".chungoid/execution.log"
  max_size: 10485760                    # 10MB
  backup_count: 5
```

### Environment Variables

Set these in your `.env` file or environment:

```bash
# Required for OpenAI
OPENAI_API_KEY=your_openai_api_key

# Required for Anthropic
ANTHROPIC_API_KEY=your_anthropic_api_key

# Optional: Custom paths
CHUNGOID_CONFIG_PATH=.chungoid/config.yaml
CHROMA_DB_PATH=.chungoid/chroma_db
CHUNGOID_LOG_LEVEL=INFO

# Optional: LLM Settings
LLM_MAX_TOKENS=4000
LLM_TEMPERATURE=0.7
```

## Writing Effective Goals

### Goal File Structure

Create clear, structured goals in your `goal.txt` file:

```text
Build a task management web application with the following features:

Core Functionality:
- User registration and authentication
- Create, edit, delete tasks
- Organize tasks by categories
- Mark tasks as complete/incomplete
- Due date reminders

Technical Requirements:
- Python Flask backend
- SQLite database
- Bootstrap frontend
- RESTful API design
- Unit tests with pytest
- Docker containerization

Non-Functional Requirements:
- Mobile-responsive design
- Load time under 2 seconds
- Support for 100 concurrent users
- GDPR compliance for user data

Out of Scope:
- Real-time collaboration
- Mobile app development
- Third-party integrations
```

### Goal Writing Best Practices

#### 1. Be Specific and Clear
```text
List libraries, structure, intention, workflow, etc. Discuss with `chungoid discuss`
for optimal goal refinement & better results.
```

#### 2. Include Technical Preferences
```text
Include technical stack preferences:
- Backend: Python Flask or FastAPI
- Database: PostgreSQL or SQLite
- Frontend: Bootstrap or Tailwind CSS
- Testing: pytest
- Deployment: Docker
```

#### 3. Define Success Criteria
```text
Success Criteria:
- All endpoints return proper HTTP status codes
- 90%+ test coverage
- Application runs without errors
- Documentation includes setup instructions
```

#### 4. Specify Constraints
```text
Constraints:
- Must work with Python 3.11+
- Database should be SQLite for simplicity
- No external dependencies beyond pip packages
- Budget: 50 API calls maximum
```

### Complex Goal Examples

#### E-commerce Platform
```text
Build a complete e-commerce platform with:

Product Management:
- Product catalog with categories
- Inventory tracking
- Image upload and storage
- Search and filtering

User Features:
- Customer registration/login
- Shopping cart functionality
- Order processing
- Payment integration (mock/Stripe)

Admin Features:
- Admin dashboard
- Order management
- Customer management
- Analytics and reporting

Technical Stack:
- FastAPI backend
- PostgreSQL database
- React frontend (or server-side templates)
- Redis for caching
- Docker deployment

Quality Requirements:
- 95% test coverage
- API documentation with OpenAPI
- Performance: <200ms response time
- Security: Input validation, SQL injection prevention
```

#### Data Analytics Pipeline
```text
Create a data analytics pipeline for processing e-commerce data:

Data Sources:
- CSV files with sales data
- JSON API for product information
- Database for customer data

Processing Steps:
- Data cleaning and validation
- Feature engineering
- Statistical analysis
- Trend identification
- Anomaly detection

Output:
- Daily/weekly/monthly reports
- Data visualizations (charts, graphs)
- Automated alerts for anomalies
- Export to Excel and PDF

Technical Requirements:
- Python with pandas, numpy, matplotlib
- Jupyter notebooks for analysis
- SQLite for data storage
- Scheduled execution with cron
- Logging and error handling
- Configuration via YAML files
```

## Agent Ecosystem

### Core Agents

#### MasterPlannerAgent
Creates high-level execution strategies from goals.

**Inputs:**
- Refined user goal
- Project constraints
- Available resources

**Outputs:**
- Master execution plan
- Stage definitions
- Resource allocation

#### ProductAnalystAgent_v1
Generates detailed requirements (LOPRD) from goals.

**Inputs:**
- Goal description
- User preferences
- Domain context

**Outputs:**
- LLM-Optimized Product Requirements Document
- User stories and acceptance criteria
- Non-functional requirements

#### BlueprintGeneratorAgent_v1
Creates architectural blueprints and technical specifications.

**Inputs:**
- LOPRD document
- Technical constraints
- Platform preferences

**Outputs:**
- System architecture diagrams
- Technology stack decisions
- Component specifications

#### EnvironmentBootstrapAgent_v1
Sets up development environments and dependencies.

**Supported Languages:**
- Python (virtualenv, conda, pip)
- Node.js (npm, yarn, nvm)
- Java (Maven, Gradle)
- Go (go mod)

**Outputs:**
- Configured virtual environment
- Installed dependencies
- Environment validation report

#### SmartCodeGeneratorAgent_v1
Generates context-aware, production-ready code.

**Features:**
- Multi-language support
- Context-aware generation
- Code style compliance
- Documentation generation
- Error handling patterns

**Inputs:**
- Code specifications
- Existing codebase context
- Style preferences

**Outputs:**
- Production-ready source code
- Documentation comments
- Configuration files

#### CoreTestGeneratorAgent_v1
Creates comprehensive test suites.

**Test Types:**
- Unit tests
- Integration tests
- End-to-end tests
- Performance tests
- Security tests

**Frameworks:**
- Python: pytest, unittest
- JavaScript: Jest, Mocha
- Java: JUnit, TestNG

### Quality Assurance Agents

#### AutomatedRefinementCoordinatorAgent_v1 (ARCA)
Orchestrates quality assurance and iterative refinement.

**Responsibilities:**
- Artifact evaluation
- Quality metrics assessment
- Refinement coordination
- Human intervention triggers

#### ProactiveRiskAssessorAgent_v1 (PRAA)
Identifies potential risks and optimization opportunities.

**Analysis Areas:**
- Security vulnerabilities
- Performance bottlenecks
- Maintainability issues
- Scalability concerns

#### RequirementsTracerAgent_v1 (RTA)
Ensures requirements traceability across artifacts.

**Tracking:**
- Requirements coverage
- Implementation completeness
- Test adequacy
- Documentation alignment

## MCP Tools

### File System Tools

#### `filesystem_project_scan`
Scans project directories for analysis.

```python
# Usage in agents
scan_result = await self.tools.filesystem_project_scan(
    project_dir="/path/to/project",
    include_patterns=["*.py", "*.js", "*.yaml"],
    exclude_patterns=["*.pyc", "__pycache__"],
    max_files=100
)
```

#### `filesystem_read_file`
Reads file contents with encoding detection.

```python
content = await self.tools.filesystem_read_file(
    file_path="/path/to/file.py",
    encoding="utf-8"
)
```

#### `filesystem_write_file`
Writes content to files with backup support.

```python
await self.tools.filesystem_write_file(
    file_path="/path/to/new_file.py",
    content=generated_code,
    backup=True,
    create_dirs=True
)
```

### ChromaDB Tools

#### `chroma_add_documents`
Stores documents in ChromaDB collections.

```python
await self.tools.chroma_add_documents(
    collection_name="project_artifacts",
    documents=[code_content],
    metadatas=[{"type": "generated_code", "language": "python"}],
    ids=["code_123"]
)
```

#### `chroma_query_documents`
Retrieves similar documents using semantic search.

```python
results = await self.tools.chroma_query_documents(
    collection_name="project_artifacts",
    query_text="authentication code",
    n_results=5,
    filter_metadata={"type": "generated_code"}
)
```

### Analysis Tools

#### `analysis_dependency_scan`
Analyzes project dependencies and security issues.

```python
deps = await self.tools.analysis_dependency_scan(
    project_dir="/path/to/project",
    include_dev_deps=True,
    security_check=True
)
```

#### `analysis_code_quality`
Performs code quality analysis.

```python
quality = await self.tools.analysis_code_quality(
    file_path="/path/to/code.py",
    style_guide="pep8",
    complexity_threshold=10
)
```

### Testing Tools

#### `testing_run_pytest`
Executes pytest test suites.

```python
test_results = await self.tools.testing_run_pytest(
    project_dir="/path/to/project",
    test_path="tests/",
    coverage=True,
    parallel=True
)
```

#### `testing_generate_test`
Generates test cases for code.

```python
tests = await self.tools.testing_generate_test(
    source_code=function_code,
    test_framework="pytest",
    coverage_target=90
)
```

## Workflows and Execution

### Standard Build Workflow

The default build process follows these stages:

1. **Goal Analysis** (`ProductAnalystAgent_v1`)
   - Refine user goals
   - Generate LOPRD
   - Identify requirements

2. **Architecture Design** (`BlueprintGeneratorAgent_v1`)
   - Create system blueprint
   - Select technology stack
   - Define component structure

3. **Planning** (`MasterPlannerAgent`)
   - Generate execution plan
   - Define stage dependencies
   - Allocate resources

4. **Environment Setup** (`EnvironmentBootstrapAgent_v1`)
   - Create virtual environment
   - Install dependencies
   - Validate setup

5. **Code Generation** (`SmartCodeGeneratorAgent_v1`)
   - Generate source code
   - Create configuration files
   - Implement core functionality

6. **Testing** (`CoreTestGeneratorAgent_v1`)
   - Generate test suites
   - Execute tests
   - Validate coverage

7. **Documentation** (`ProjectDocumentationAgent_v1`)
   - Generate README
   - Create API documentation
   - Write usage guides

### Custom Workflows

Define custom workflows using YAML:

```yaml
name: "Microservice Development"
version: "1.0"
description: "Custom workflow for microservice development"
start_stage: "requirements"

stages:
  requirements:
    agent_id: "ProductAnalystAgent_v1"
    inputs:
      goal_file: "goal.txt"
    next: "architecture"
    success_criteria:
      - "LOPRD generated"
      - "Requirements validated"
    
  architecture:
    agent_id: "BlueprintGeneratorAgent_v1"
    inputs:
      loprd_doc_id: "{{ outputs.requirements.loprd_doc_id }}"
    next: 
      condition: "outputs.architecture.complexity > 5"
      true: "detailed_design"
      false: "implementation"
    
  detailed_design:
    agent_id: "DetailedDesignAgent_v1"
    next: "implementation"
    
  implementation:
    parallel_group: "development"
    stages:
      - id: "backend"
        agent_id: "SmartCodeGeneratorAgent_v1"
        inputs:
          component: "backend"
      - id: "tests"
        agent_id: "CoreTestGeneratorAgent_v1"
        depends_on: ["backend"]
    
  deployment:
    agent_id: "DeploymentAgent_v1"
    depends_on: ["development"]
    inputs:
      target_platform: "docker"
```

### Conditional Execution

Use conditions to create dynamic workflows:

```yaml
stages:
  code_review:
    agent_id: "CodeReviewAgent_v1"
    next:
      condition: "outputs.code_review.quality_score < 0.8"
      true: "code_refinement"
      false: "testing"
      
  code_refinement:
    agent_id: "CodeRefinementAgent_v1"
    next: "code_review"  # Loop back for re-review
    max_iterations: 3
```

### Parallel Execution

Execute stages in parallel for efficiency:

```yaml
stages:
  frontend:
    agent_id: "FrontendGeneratorAgent_v1"
    parallel_group: "ui_development"
    
  backend:
    agent_id: "BackendGeneratorAgent_v1"
    parallel_group: "ui_development"
    
  database:
    agent_id: "DatabaseDesignAgent_v1"
    parallel_group: "ui_development"
    
  integration:
    agent_id: "IntegrationAgent_v1"
    depends_on: ["frontend", "backend", "database"]
```

## Human-in-the-Loop

### Pause Points

Configure automatic pause points for human review:

```yaml
# In config.yaml
execution:
  human_intervention_points: 
    - "blueprint"      # Pause after architecture design
    - "code"          # Pause after code generation
    - "testing"       # Pause after test execution
```

Or use CLI flags:

```bash
chungoid build --goal-file goal.txt --pause-after blueprint
```

### Manual Review Process

1. **Review Generated Artifacts**
   ```bash
   # Review the blueprint
   cat .chungoid/artifacts/project_blueprint.md
   
   # Review the requirements
   cat .chungoid/artifacts/llm_optimized_prd.json
   ```

2. **Provide Feedback**
   ```bash
   # Create feedback file
   echo "The authentication system needs two-factor authentication support" > .chungoid/feedback.txt
   ```

3. **Resume or Modify**
   ```bash
   # Resume with feedback
   chungoid resume --project-dir . --feedback-file .chungoid/feedback.txt
   
   # Or restart with modifications
   chungoid build --goal-file modified_goal.txt --project-dir . --restart-from blueprint
   ```

### Interactive Mode

Use interactive mode for complex requirements:

```bash
chungoid interactive --goal "Build a social media platform"
```

This launches a conversational interface where agents ask clarifying questions:

```
🤖 ProductAnalyst: I need to clarify some requirements for your social media platform.

What type of content should users be able to share?
1. Text posts only
2. Text and images
3. Text, images, and videos
4. All media types including documents

User: 2

🤖 ProductAnalyst: Great! Should the platform include:
- User profiles and following system?
- Private messaging?
- Content moderation features?
- Analytics for users?

Please provide details for each feature you want included.
```

## Monitoring and Debugging

### Execution Monitoring

Monitor execution in real-time:

```bash
# Watch execution progress
chungoid status --project-dir . --watch

# View detailed logs
tail -f .chungoid/execution.log

# Monitor agent activity
chungoid monitor --project-dir . --agent-filter CodeGeneratorAgent
```

### Debug Mode

Enable debug mode for detailed logging:

```bash
chungoid build --goal-file goal.txt --debug --project-dir .
```

Debug mode provides:
- Detailed agent reasoning
- Tool invocation traces
- Performance metrics
- Error stack traces

### Execution State

Check current execution state:

```bash
chungoid status --project-dir .
```

Output example:
```
Project Status: PAUSED_FOR_REVIEW
Current Stage: code_generation
Progress: 65% complete
Active Agents: SmartCodeGeneratorAgent_v1
Last Activity: 2024-01-15 14:30:22

Completed Stages:
✓ requirements (ProductAnalystAgent_v1) - 95% confidence
✓ architecture (BlueprintGeneratorAgent_v1) - 88% confidence
✓ environment (EnvironmentBootstrapAgent_v1) - 100% confidence
◐ code_generation (SmartCodeGeneratorAgent_v1) - In Progress

Pending Stages:
- testing (CoreTestGeneratorAgent_v1)
- documentation (ProjectDocumentationAgent_v1)
```

### Error Handling

Common error patterns and solutions:

#### Agent Timeout
```
Error: AgentTimeoutError - SmartCodeGeneratorAgent_v1 exceeded 300s timeout
```

**Solution:**
```yaml
# Increase timeout in config.yaml
agents:
  timeout: 600
  SmartCodeGeneratorAgent_v1:
    timeout: 900  # Agent-specific override
```

#### LLM Rate Limiting
```
Error: RateLimitError - API rate limit exceeded
```

**Solution:**
```yaml
# Add retry configuration
llm:
  rate_limit_retry: true
  retry_delay: 60
  max_retries: 5
```

#### ChromaDB Connection Issues
```
Error: ChromaConnectionError - Failed to connect to ChromaDB
```

**Solution:**
```bash
# Reset ChromaDB
rm -rf .chungoid/chroma_db
chungoid init --reset-chroma
```

### Performance Monitoring

Track performance metrics:

```bash
# View execution metrics
chungoid metrics --project-dir .

# Export metrics
chungoid metrics --project-dir . --export metrics.json
```

Example metrics output:
```json
{
  "execution_time": "1247.5s",
  "agent_performance": {
    "ProductAnalystAgent_v1": {
      "execution_time": "45.2s",
      "confidence_score": 0.95,
      "llm_calls": 3,
      "tokens_used": 2847
    },
    "SmartCodeGeneratorAgent_v1": {
      "execution_time": "423.8s",
      "confidence_score": 0.89,
      "llm_calls": 12,
      "tokens_used": 15692
    }
  },
  "resource_usage": {
    "peak_memory": "2.3GB",
    "cpu_time": "892.1s",
    "disk_space": "156MB"
  }
}
```

## Best Practices

### Goal Writing

1. **Start Simple**: Begin with basic functionality and iterate
2. **Be Specific**: Include exact technical requirements
3. **Define Success**: Clear acceptance criteria
4. **Consider Constraints**: Budget, time, and resource limits

### Project Organization

1. **Use Version Control**: Initialize git repository before building
2. **Backup Artifacts**: Keep copies of important generated artifacts
3. **Document Changes**: Track modifications and iterations
4. **Test Early**: Validate generated code frequently

### Configuration Management

1. **Environment Separation**: Use different configs for dev/prod
2. **Secret Management**: Never commit API keys
3. **Validation**: Test configuration changes in isolated environments
4. **Documentation**: Document custom configuration choices

### Performance Optimization

1. **Parallel Execution**: Enable parallel stages when possible
2. **Resource Limits**: Set appropriate timeouts and retry limits
3. **Caching**: Use ChromaDB caching for repeated queries
4. **Monitoring**: Track performance metrics and optimize bottlenecks

### Quality Assurance

1. **Human Review**: Pause at critical milestones
2. **Iterative Refinement**: Use feedback loops for improvement
3. **Automated Testing**: Ensure comprehensive test coverage
4. **Code Quality**: Configure style guides and quality thresholds

### Security Considerations

1. **API Key Security**: Use environment variables for secrets
2. **Input Validation**: Validate all user inputs and goals
3. **Dependency Security**: Enable security scanning for dependencies
4. **Access Control**: Limit file system access appropriately

### Troubleshooting Workflow

1. **Check Logs**: Always start with execution logs
2. **Verify Configuration**: Ensure all settings are correct
3. **Test Components**: Isolate and test individual agents
4. **Reset State**: Clear ChromaDB and restart if needed
5. **Seek Help**: Use debug mode and community resources
