# Chungoid Configuration Guide

This guide provides comprehensive documentation for configuring the Chungoid autonomous development system. Understanding configuration is crucial for optimizing performance, customizing behavior, and ensuring proper system operation.

## Table of Contents

- [Configuration Overview](#configuration-overview)
- [Configuration Hierarchy](#configuration-hierarchy)
- [Configuration Sections](#configuration-sections)
- [Environment Variables](#environment-variables)
- [Configuration Examples](#configuration-examples)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)
- [Troubleshooting](#troubleshooting)

## Configuration Overview

Chungoid uses a hierarchical configuration system built on Pydantic models, providing type safety, validation, and flexible configuration sources. The system supports:

- **Hierarchical Precedence**: Environment variables override files, which override defaults
- **Type Safety**: All configuration values are validated using Pydantic models
- **Secure Secrets**: Sensitive data is handled only through environment variables
- **Runtime Updates**: Configuration can be updated without system restarts
- **Performance Optimization**: Settings for different deployment scenarios

## Configuration Hierarchy

Configuration values are resolved using the following precedence (highest to lowest):

1. **Environment Variables** (`CHUNGOID_*` prefix) - Secrets and overrides
2. **Project Configuration** (`.chungoid/config.yaml` in project root) - Project-specific settings
3. **Global Configuration** (`~/.chungoid/config.yaml`) - User-wide defaults
4. **Hardcoded Defaults** - Fallback values defined in Pydantic models

## Configuration Sections

### LLM Configuration

Controls Large Language Model providers and behavior.

```yaml
llm:
  provider: "openai"                          # LLM provider (openai, anthropic, etc.)
  default_model: "gpt-4o-mini-2024-07-18"    # Default model to use
  fallback_model: "gpt-3.5-turbo"            # Fallback if default fails
  timeout: 60                                 # Request timeout in seconds (1-600)
  max_retries: 3                              # Maximum retry attempts (0-10)
  retry_delay: 1.0                            # Delay between retries (0.1-30.0)
  rate_limit_rpm: 60                          # Rate limit requests per minute
  max_tokens_per_request: 4000                # Maximum tokens per request
  enable_cost_tracking: true                  # Track API usage costs
  monthly_budget_limit: 100.0                 # Optional monthly budget limit (USD)
```

**Key Settings:**
- `provider`: Supported providers include `openai`, `anthropic`, `ollama`, `azure`
- `api_key`: **Must be set via environment variables only** (see Environment Variables section)
- `timeout`: Balance between responsiveness and allowing complex operations
- `max_tokens_per_request`: Adjust based on your use case complexity

### ChromaDB Configuration

Controls vector database connections and behavior.

```yaml
chromadb:
  host: "localhost"                           # ChromaDB host
  port: 8000                                  # ChromaDB port (1-65535)
  use_ssl: false                              # Use SSL connection
  default_collection_prefix: "chungoid"       # Collection name prefix
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
  connection_timeout: 30                      # Connection timeout (1-300)
  query_timeout: 60                           # Query timeout (1-600)
  max_batch_size: 100                         # Batch size for operations (1-1000)
  auto_cleanup_enabled: true                  # Enable automatic cleanup
  cleanup_retention_days: 30                  # Retention period for cleanup
```

**Key Settings:**
- `auth_token`: **Must be set via environment variables only** for secure deployments
- `embedding_model`: Choose based on your language and performance requirements
- `max_batch_size`: Increase for better performance with large datasets

### Project Configuration

Controls project-specific behavior and analysis.

```yaml
project:
  name: "my_project"                          # Optional project name
  project_type: "python"                     # Optional project type hint
  auto_detect_dependencies: true             # Automatically detect dependencies
  auto_detect_project_type: true             # Automatically detect project type
  enable_smart_caching: true                 # Enable intelligent caching
  analysis_depth: 5                          # Directory depth for analysis (1-20)
  enable_content_analysis: true              # Enable file content analysis
  max_file_size_mb: 10                       # Maximum file size to analyze (1-100)
  include_hidden_files: false                # Include hidden files in analysis
  exclude_patterns:                          # File patterns to exclude
    - "__pycache__"
    - "node_modules"
    - ".git"
    - "*.pyc"
    - "*.log"
  preferred_language_models: []              # Preferred models for this project
```

### Agent Configuration

Controls agent execution behavior and performance.

```yaml
agents:
  # Execution Settings
  default_timeout: 300                       # Default agent timeout (30-3600)
  max_concurrent_agents: 5                   # Maximum concurrent agents (1-20)
  enable_parallel_execution: true            # Enable parallel execution
  default_max_iterations: 10                 # Override hardcoded iteration limits (1-100)
  
  # Stage-specific max iterations
  stage_max_iterations:
    environment_bootstrap: 5                 # Bootstrap stage iterations
    dependency_management: 8                 # Dependency management iterations
    enhanced_architecture_design: 15         # Architecture design iterations
    risk_assessment: 10                      # Risk assessment iterations
    code_generation: 25                      # Code generation iterations
    code_debugging: 12                       # Debugging iterations
    product_analysis: 8                      # Product analysis iterations
    requirements_tracing: 6                  # Requirements tracing iterations
    project_documentation: 10                # Documentation iterations
    automated_refinement: 8                  # Refinement iterations
  
  # Retry and Resilience
  max_retries: 3                             # Maximum retry attempts (0-10)
  retry_exponential_backoff: true            # Use exponential backoff
  base_retry_delay: 2.0                      # Base retry delay (0.1-60.0)
  
  # Checkpointing
  enable_automatic_checkpoints: true         # Enable checkpoints
  checkpoint_frequency: 5                    # Checkpoint every N stages (1-50)
  enable_state_compression: true             # Compress large states
  
  # Monitoring
  enable_performance_monitoring: true        # Enable performance monitoring
  log_agent_outputs: true                    # Log detailed outputs
  enable_health_checks: true                 # Enable health monitoring
  
  # Agent-specific overrides
  agent_timeouts:
    "EnvironmentBootstrapAgent": 300
    "DependencyManagementAgent_v1": 600
    "EnhancedArchitectAgent_v1": 1200
    "SmartCodeGeneratorAgent_v1": 1800
  
  agent_retry_limits:
    "EnvironmentBootstrapAgent": 2
    "SmartCodeGeneratorAgent_v1": 3
```

### Logging Configuration

Controls logging behavior and output.

```yaml
logging:
  level: "INFO"                              # Log level (DEBUG, INFO, WARNING, ERROR)
  enable_file_logging: true                  # Enable file logging
  log_directory: "logs"                      # Log directory path
  enable_structured_logging: true            # Use JSON structured logging
  include_timestamps: true                   # Include timestamps
  include_caller_info: false                 # Include caller information
  max_log_size_mb: 100                       # Max log file size (1-1000)
  log_retention_days: 7                      # Log retention period (1-365)
  compress_old_logs: true                    # Compress rotated logs
  mask_secrets: true                         # Mask secrets in logs
  log_api_requests: false                    # Log API request details
  enable_debug_logging: false                # Enable debug logging
```

### System Configuration

Controls system-wide behavior.

```yaml
config_version: "1.0.0"                     # Configuration schema version
environment: "development"                  # Runtime environment
enable_telemetry: false                     # Enable anonymous telemetry
data_directory: "~/.chungoid"               # Data directory path
```

## Environment Variables

Environment variables provide the highest precedence and are the **only** way to set secrets. All environment variables use the `CHUNGOID_` prefix.

### LLM Provider Secrets

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."
export CHUNGOID_OPENAI_API_KEY="sk-..."    # Alternative prefix

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."
export CHUNGOID_ANTHROPIC_API_KEY="sk-ant-..."

# General LLM settings
export CHUNGOID_LLM_PROVIDER="openai"
export CHUNGOID_LLM_DEFAULT_MODEL="gpt-4o-mini-2024-07-18"
export CHUNGOID_LLM_TIMEOUT="120"
export CHUNGOID_LLM_MAX_RETRIES="2"
```

### ChromaDB Authentication

```bash
export CHUNGOID_CHROMADB_HOST="localhost"
export CHUNGOID_CHROMADB_PORT="8000"
export CHUNGOID_CHROMADB_AUTH_TOKEN="your-auth-token"
export CHUNGOID_CHROMADB_USE_SSL="true"
```

### Agent Configuration

```bash
export CHUNGOID_AGENT_TIMEOUT="300"
export CHUNGOID_AGENT_MAX_CONCURRENT="3"
export CHUNGOID_AGENT_MAX_RETRIES="2"
export CHUNGOID_MAX_ITERATIONS="15"        # Global override for all agents
```

### System Settings

```bash
export CHUNGOID_ENVIRONMENT="production"
export CHUNGOID_DATA_DIRECTORY="/opt/chungoid"
export CHUNGOID_LOG_LEVEL="WARNING"
export CHUNGOID_ENABLE_DEBUG="false"
```

## Configuration Examples

### Development Configuration

Standard development setup with full features enabled:

```yaml
# .chungoid/config.yaml
config_version: "1.0.0"
environment: "development"

llm:
  provider: "openai"
  default_model: "gpt-4o-mini-2024-07-18"
  timeout: 60
  max_retries: 3
  enable_cost_tracking: true

chromadb:
  host: "localhost"
  port: 8000
  auto_cleanup_enabled: true
  cleanup_retention_days: 7

agents:
  default_timeout: 600
  max_concurrent_agents: 5
  enable_parallel_execution: true
  stage_max_iterations:
    environment_bootstrap: 8
    dependency_management: 10
    enhanced_architecture_design: 20
    code_generation: 30
    code_debugging: 15

logging:
  level: "INFO"
  enable_file_logging: true
  enable_structured_logging: true
```

### Fast Testing Configuration

Optimized for rapid iteration and testing (similar to the example_config.yaml created earlier):

```yaml
# config/fast-testing.yaml
config_version: "1.0.0"
environment: "fast_testing"

llm:
  provider: "openai"
  default_model: "gpt-4o-mini-2024-07-18"
  timeout: 30                                # Ultra-fast timeouts
  max_retries: 1                             # Fail fast
  max_tokens_per_request: 4000

agents:
  default_timeout: 180                       # 3 minutes max per agent
  max_concurrent_agents: 2
  enable_parallel_execution: false           # Sequential for stability
  max_retries: 1                             # Immediate failure feedback
  
  stage_max_iterations:
    environment_bootstrap: 3                 # Minimal setup
    dependency_management: 3
    enhanced_architecture_design: 8          # Basic blueprint
    risk_assessment: 4
    code_generation: 15                      # Focused generation
    code_debugging: 8
    product_analysis: 1                      # Skip for testing
    requirements_tracing: 1
    project_documentation: 1
    automated_refinement: 1

chromadb:
  enabled: false                             # Disable for speed

logging:
  level: "WARNING"                           # Only errors and warnings
  enable_file_logging: false                # No file logging
  include_timestamps: false
```

### Production Configuration

Optimized for stability and monitoring:

```yaml
# .chungoid/config.yaml
config_version: "1.0.0"
environment: "production"

llm:
  provider: "openai"
  default_model: "gpt-4o-mini-2024-07-18"
  timeout: 120
  max_retries: 5
  rate_limit_rpm: 30                         # Conservative rate limiting
  monthly_budget_limit: 500.0

chromadb:
  use_ssl: true
  connection_timeout: 60
  auto_cleanup_enabled: true
  cleanup_retention_days: 90

agents:
  default_timeout: 900                       # Longer timeouts for stability
  max_concurrent_agents: 3                   # Conservative concurrency
  enable_automatic_checkpoints: true
  checkpoint_frequency: 3                    # More frequent checkpoints
  enable_performance_monitoring: true

logging:
  level: "INFO"
  enable_file_logging: true
  enable_structured_logging: true
  log_retention_days: 30
  mask_secrets: true
```

## Performance Optimization

### Speed Optimization Strategies

**For Fast Testing/Development:**
- Reduce `timeout` values (30-60s for LLM, 180-300s for agents)
- Lower `stage_max_iterations` (3-15 per stage)
- Set `max_retries` to 0-1 for immediate feedback
- Disable ChromaDB (`chromadb.enabled: false`)
- Use minimal logging (`level: "WARNING"`, `enable_file_logging: false`)
- Disable parallel execution for predictable testing

**Example Fast Config Pattern:**
```yaml
llm:
  timeout: 30
  max_retries: 1
agents:
  default_timeout: 180
  max_retries: 1
  stage_max_iterations:
    code_generation: 10    # Reduce from default 25-50
logging:
  level: "WARNING"
  enable_file_logging: false
```

### Quality Optimization Strategies

**For Production/High-Quality Output:**
- Increase `timeout` values (120-300s for LLM, 600-1800s for agents)
- Higher `stage_max_iterations` (15-50 per stage)
- More `max_retries` (3-5) for resilience
- Enable all monitoring and checkpointing
- Use structured logging for debugging

**Example Quality Config Pattern:**
```yaml
llm:
  timeout: 120
  max_retries: 5
agents:
  default_timeout: 900
  max_retries: 3
  stage_max_iterations:
    enhanced_architecture_design: 25
    code_generation: 50
  enable_automatic_checkpoints: true
logging:
  level: "INFO"
  enable_structured_logging: true
```

### Memory Optimization

- Reduce `chromadb.max_batch_size` for memory-constrained environments
- Enable `agents.enable_state_compression` for large projects
- Lower `agents.max_concurrent_agents` to reduce memory usage
- Set appropriate `project.max_file_size_mb` limits

## Security Best Practices

### Secret Management

1. **Never put secrets in configuration files**
2. **Always use environment variables for API keys and tokens**
3. **Enable `logging.mask_secrets: true` in all environments**
4. **Use minimal permissions for service accounts**

```bash
# Correct way to set secrets
export OPENAI_API_KEY="sk-..."
export CHUNGOID_CHROMADB_AUTH_TOKEN="token"

# Never do this in config files:
# llm:
#   api_key: "sk-..."  # ❌ NEVER DO THIS
```

### Production Security

```yaml
logging:
  mask_secrets: true                         # Always enable
  log_api_requests: false                    # Disable in production
  enable_debug_logging: false                # Disable debug logs

chromadb:
  use_ssl: true                              # Enable SSL in production
  
system:
  enable_telemetry: false                    # Disable if privacy required
```

## Troubleshooting

### Common Configuration Issues

**1. API Key Not Found**
```
Error: OpenAI API key not found
```
**Solution:** Set the environment variable correctly:
```bash
export OPENAI_API_KEY="sk-your-key"
# or
export CHUNGOID_LLM_API_KEY="sk-your-key"
```

**2. ChromaDB Connection Failed**
```
Error: Failed to connect to ChromaDB
```
**Solution:** Check ChromaDB service and configuration:
```bash
# Check if ChromaDB is running
curl http://localhost:8000/api/v1/heartbeat

# Update configuration
export CHUNGOID_CHROMADB_HOST="correct-host"
export CHUNGOID_CHROMADB_PORT="8000"
```

**3. Agent Timeout Errors**
```
Error: Agent execution timed out
```
**Solution:** Increase timeout values:
```yaml
agents:
  default_timeout: 900  # Increase from default
  agent_timeouts:
    "SmartCodeGeneratorAgent_v1": 1800  # Longer for code generation
```

**4. Too Many Iterations**
```
Warning: Agent reached maximum iterations
```
**Solution:** Adjust iteration limits:
```yaml
agents:
  stage_max_iterations:
    code_generation: 50  # Increase for complex tasks
```

### Configuration Validation

Use the CLI to validate your configuration:

```bash
# Check current configuration
chungoid config show

# Validate configuration file
chungoid config validate --config-file .chungoid/config.yaml

# Test LLM connection
chungoid config test-llm

# Test ChromaDB connection  
chungoid config test-chromadb
```

### Getting Help

1. **Check the logs** in the configured log directory
2. **Enable debug logging** temporarily:
   ```yaml
   logging:
     level: "DEBUG"
     enable_debug_logging: true
   ```
3. **Validate your configuration** using the CLI tools
4. **Check environment variables** are set correctly
5. **Review the configuration hierarchy** to understand precedence

For more specific guidance, see the [LiteLLM Setup Guide](litellm_setup.md) for LLM provider configuration details. 