# Configuration Reference

This document lists ALL available options from the configuration schema.

## Complete Configuration Options

Based on the `SystemConfiguration` Pydantic model in `config_manager.py`, here are ALL available configuration options:

```yaml
# ============================================================================
# Configuration Metadata
# ============================================================================
config_version: "1.0.0"                     # Configuration schema version  
last_updated: "2025-01-20T10:00:00Z"        # Last configuration update timestamp

# ============================================================================
# LLM Configuration
# ============================================================================
llm:
  # Provider settings
  provider: "openai"                         # LLM provider (openai, anthropic, etc.)
  api_key: null                             # API key (from environment only - CHUNGOID_LLM_API_KEY)
  api_base_url: null                        # Custom API base URL
  
  # Model preferences  
  default_model: "gpt-4o-mini-2024-07-18"  # Default model to use
  fallback_model: "gpt-3.5-turbo"          # Fallback model if default fails
  
  # Performance settings
  timeout: 60                               # Request timeout in seconds (1-600)
  max_retries: 3                           # Maximum retry attempts (0-10)
  retry_delay: 1.0                         # Delay between retries (0.1-30.0)
  rate_limit_rpm: 60                       # Rate limit requests per minute (>=1)
  
  # Cost management
  max_tokens_per_request: 4000             # Maximum tokens per request (>=1)
  enable_cost_tracking: true               # Track API usage costs
  monthly_budget_limit: null               # Monthly budget limit in USD (>=0)

# ============================================================================
# ChromaDB Configuration  
# ============================================================================
chromadb:
  # Connection settings
  host: "localhost"                        # ChromaDB host
  port: 8000                              # ChromaDB port (1-65535)
  auth_token: null                        # Authentication token (from environment only)
  use_ssl: false                          # Use SSL connection
  
  # Database settings
  default_collection_prefix: "chungoid"   # Default collection name prefix
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"  # Embedding model
  
  # Performance settings
  connection_timeout: 30                  # Connection timeout in seconds (1-300)
  query_timeout: 60                       # Query timeout in seconds (1-600)
  max_batch_size: 100                     # Maximum batch size for operations (1-1000)
  
  # Data management
  auto_cleanup_enabled: true              # Enable automatic cleanup of old data
  cleanup_retention_days: 30              # Retention period for cleanup (>=1)

# ============================================================================
# Project Configuration
# ============================================================================
project:
  # Project identification
  name: null                            # Project name
  project_type: null                    # Project type (python, javascript, etc.)
  description: null                     # Project description
  
  # Project behavior
  auto_detect_dependencies: true        # Automatically detect dependencies
  auto_detect_project_type: true        # Automatically detect project type
  enable_smart_caching: true            # Enable intelligent caching
  
  # File and directory settings
  exclude_patterns:                     # File patterns to exclude from analysis
    - "__pycache__"
    - "node_modules"
    - ".git"
    - "*.pyc"
    - "*.log"
  include_hidden_files: false           # Include hidden files in analysis
  max_file_size_mb: 10                  # Maximum file size to analyze (1-100 MB)
  
  # Analysis preferences
  analysis_depth: 5                     # Maximum directory depth for analysis (1-20)
  enable_content_analysis: true         # Enable file content analysis
  preferred_language_models: []         # Preferred language models for this project

# ============================================================================
# Agent Configuration
# ============================================================================
agents:
  # Execution settings
  default_timeout: 300                 # Default agent timeout in seconds (30-3600)
  max_concurrent_agents: 5             # Maximum concurrent agents (1-20)
  enable_parallel_execution: true      # Enable parallel agent execution
  default_max_iterations: null         # Default max iterations for all agents (1-100)
  
  # Stage-specific max iterations overrides
  stage_max_iterations:                # Per-stage max iteration overrides
    product_analysis: 20               # Example stage override
    architecture_design: 25            # Example stage override
    blueprint_review: 15               # Example stage override
    code_generation: 30                # Example stage override
    project_documentation: 16          # Example stage override
    code_debugging: 28                 # Example stage override
    automated_refinement: 35           # Example stage override
    environment_bootstrap: 15          # Example stage override
    dependency_management: 12          # Example stage override
    requirements_tracing: 18           # Example stage override
    risk_assessment: 22                # Example stage override
  
  # Retry and resilience
  max_retries: 3                       # Maximum retry attempts for failed agents (0-10)
  retry_exponential_backoff: true      # Use exponential backoff for retries
  base_retry_delay: 2.0                # Base retry delay in seconds (0.1-60.0)
  
  # Checkpoint and state management
  enable_automatic_checkpoints: true   # Enable automatic checkpoint creation
  checkpoint_frequency: 5              # Checkpoint frequency (every N stages) (1-50)
  enable_state_compression: true       # Compress large execution states
  
  # Monitoring and observability
  enable_performance_monitoring: true  # Enable agent performance monitoring
  log_agent_outputs: true              # Log detailed agent outputs
  enable_health_checks: true           # Enable agent health monitoring
  
  # Agent-specific overrides
  agent_timeouts: {}                   # Per-agent timeout overrides
  agent_retry_limits: {}               # Per-agent retry limit overrides

# ============================================================================
# Logging Configuration
# ============================================================================
logging:
  # Basic logging settings
  level: "INFO"                        # Default log level
  enable_file_logging: true            # Enable logging to files
  log_directory: "logs"                # Log file directory
  
  # Log formatting
  enable_structured_logging: true      # Use structured JSON logging
  include_timestamps: true             # Include timestamps in logs
  include_caller_info: false           # Include caller information
  
  # Log rotation and retention
  max_log_size_mb: 100                 # Maximum log file size (1-1000 MB)
  log_retention_days: 7                # Log retention period (1-365)
  compress_old_logs: true              # Compress rotated log files
  
  # Security and privacy
  mask_secrets: true                   # Mask secrets in log output
  log_api_requests: false              # Log API request details
  enable_debug_logging: false          # Enable debug-level logging

# ============================================================================
# System-wide Settings  
# ============================================================================
environment: "development"             # Runtime environment
enable_telemetry: false                # Enable anonymous telemetry
data_directory: "~/.chungoid"          # Data directory path
```

## Available Stage IDs for `stage_max_iterations`

These are the stage IDs that can be configured:

- `product_analysis` - Requirements and scope analysis
- `architecture_design` - System architecture design  
- `blueprint_review` - Architecture review and validation
- `code_generation` - Code file generation
- `project_documentation` - Documentation generation
- `code_debugging` - Code analysis and debugging
- `automated_refinement` - Final quality assurance and refinement
- `environment_bootstrap` - Environment setup
- `dependency_management` - Dependency analysis and installation
- `requirements_tracing` - Requirements traceability
- `risk_assessment` - Project risk assessment

### Example: Setting Stage-Specific Max Iterations

```yaml
agents:
  # Global default for all stages (if not overridden)
  default_max_iterations: 10   # applies one value to all agents, not particularly optimal
  
  # Override max iterations for specific stages (preferred method)
  stage_max_iterations:
    code_generation: 25        # Allow more iterations for complex code generation
    architecture_design: 15    # Thorough architecture planning
    code_debugging: 20         # Extensive debugging and optimization
    environment_bootstrap: 5   # Simple environment setup
```

## Environment Variable Mappings

The following environment variables can override configuration values:

### LLM Configuration
- `CHUNGOID_LLM_PROVIDER` → `llm.provider`
- `CHUNGOID_LLM_API_KEY` → `llm.api_key`
- `CHUNGOID_LLM_API_BASE_URL` → `llm.api_base_url`
- `CHUNGOID_LLM_DEFAULT_MODEL` → `llm.default_model`
- `CHUNGOID_LLM_TIMEOUT` → `llm.timeout`
- `CHUNGOID_LLM_MAX_RETRIES` → `llm.max_retries`
- `ANTHROPIC_API_KEY` → `llm.api_key` (common Anthropic env var)
- `OPENAI_API_KEY` → `llm.api_key` (common OpenAI env var)

### ChromaDB Configuration
- `CHUNGOID_CHROMADB_HOST` → `chromadb.host`
- `CHUNGOID_CHROMADB_PORT` → `chromadb.port`
- `CHUNGOID_CHROMADB_AUTH_TOKEN` → `chromadb.auth_token`
- `CHUNGOID_CHROMADB_USE_SSL` → `chromadb.use_ssl`

### Agent Configuration
- `CHUNGOID_AGENT_TIMEOUT` → `agents.default_timeout`
- `CHUNGOID_AGENT_MAX_CONCURRENT` → `agents.max_concurrent_agents`
- `CHUNGOID_AGENT_MAX_RETRIES` → `agents.max_retries`
- `CHUNGOID_MAX_ITERATIONS` → `agents.default_max_iterations`

### System Configuration
- `CHUNGOID_ENVIRONMENT` → `environment`
- `CHUNGOID_DATA_DIRECTORY` → `data_directory`
- `CHUNGOID_LOG_LEVEL` → `logging.level`
- `CHUNGOID_ENABLE_DEBUG` → `logging.enable_debug_logging`

## Configuration Precedence

Configuration values are resolved in the following order (highest to lowest precedence):

1. **Environment Variables** (CHUNGOID_* prefix) - for secrets and overrides
2. **Project Configuration** (`{project}/.chungoid/config.yaml`) - project-specific settings
3. **Global Configuration** (`~/.chungoid/config.yaml`) - user-wide defaults
4. **Hardcoded Defaults** - fallback values defined in Pydantic models

## Validation Rules

All configuration options include validation constraints:

- Numeric ranges are enforced (e.g., `timeout: 60` must be between 1-600)
- String formats are validated where applicable
- Lists and dictionaries have type checking
- Secret fields (API keys, tokens) are automatically masked in logs
- Invalid configurations will raise `ConfigurationValidationError`

## Notes

- All secret fields (API keys, auth tokens) should be set via environment variables only (.env or export VAR)
- The `stage_max_iterations` dictionary can contain any stage ID, but only the listed ones are actually used
- Agent-specific timeouts and retry limits use agent IDs as keys
- File size limits are in megabytes, timeouts in seconds
- Boolean values accept common formats: `true/false`, `yes/no`, `1/0` 