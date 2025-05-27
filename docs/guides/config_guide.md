# Chungoid Configuration Guide

This document describes **every** configurable option available to end-users via
`config.yaml` (project-local or global) and relevant environment variables. It
supersedes the older *litellm_setup.md*.

**Note on Configuration Structure:** Chungoid loads its configuration into a main
structure with top-level keys like `llm:`, `agents:`, `logging:`, and `project:`.
This guide explains user-facing aspects of these. Refer to
`chungoid-core/src/chungoid/utils/config_manager.py` for the definitive Pydantic
models (`SystemConfiguration`, `LLMConfiguration`, `AgentConfiguration`, etc.)
that define the actual loaded structure.

---

## 1. Agent Execution Settings (`agents`)

This section covers settings related to agent execution behavior, primarily managed
under the `agents:` key in your `config.yaml`. These settings influence the
`ExecutionConfig` used by agents.

```yaml
agents:
  # Default max iterations for all agents if not overridden by stage-specific config.
  # 1 means a single pass.
  default_max_iterations: 5             # (int)

  # Optional: Per-stage max iteration overrides
  # stage_max_iterations:
  #   some_specific_stage_id: 10
  #   another_stage_id: 3

  # Default agent timeout in seconds
  default_timeout: 300                  # (int) 30-3600

  # Maximum concurrent agents for parallel execution
  max_concurrent_agents: 5              # (int) 1-20

  # Enable parallel agent execution where appropriate
  enable_parallel_execution: true       # (bool)

  # Maximum retry attempts for failed agents
  max_retries: 3                        # (int) 0-10

  # Enable agent performance monitoring (e.g., metrics)
  enable_performance_monitoring: true   # (bool)

  # Other agent-related settings from config_manager.AgentConfiguration include:
  # retry_exponential_backoff, base_retry_delay,
  # enable_automatic_checkpoints, checkpoint_frequency,
  # enable_state_compression, log_agent_outputs, enable_health_checks,
  # agent_timeouts (per-agent override), agent_retry_limits (per-agent override)
```

**Conceptual Mapping to `ExecutionConfig`:**
The `agents.default_max_iterations` (and stage-specific overrides) directly
influences the `max_iterations` field in the `ExecutionConfig` object
(defined in `chungoid.schemas.unified_execution_schemas.py`) used during an
agent's execution.

Other concepts previously associated with a `uaei` block in older documentation,
such as `execution_mode`, `min_quality_score`, `stop_on_success`, and
`quality_threshold`, are attributes of the `ExecutionConfig` and
`CompletionCriteria` Pydantic models. Their behavior is determined at runtime
based on agent capabilities, the task, and the configured `max_iterations`,
rather than being direct YAML keys under the `agents:` block.

### Environment Variables
| Var                       | Maps to                         | Example                                   |
|---------------------------|---------------------------------|-------------------------------------------|
| `CHUNGOID_MAX_ITERATIONS` | `agents.default_max_iterations` | `export CHUNGOID_MAX_ITERATIONS=10`       |
| `CHUNGOID_AGENT_TIMEOUT`  | `agents.default_timeout`        | `export CHUNGOID_AGENT_TIMEOUT=600`       |
| `CHUNGOID_AGENT_MAX_CONCURRENT` | `agents.max_concurrent_agents` | `export CHUNGOID_AGENT_MAX_CONCURRENT=2` |
| `CHUNGOID_AGENT_MAX_RETRIES` | `agents.max_retries`         | `export CHUNGOID_AGENT_MAX_RETRIES=5`     |

(See `config_manager.py` for a more comprehensive list of `CHUNGOID_` env vars)

---

## 2. LLM Provider & Prompt Settings (`llm`)

Settings for Large Language Model providers, primarily via LiteLLM.

```yaml
llm:
  provider: openai                      # openai | anthropic | ollama | litellm | mock
  default_model: gpt-4o-mini-2024-07-18 # String recognised by LiteLLM
  fallback_model: gpt-3.5-turbo         # Used if primary fails
  api_key: null                         # Prefer env vars e.g. OPENAI_API_KEY
  api_base_url: null                    # Custom endpoints / Ollama URL
  timeout: 60                           # seconds
  max_retries: 3                        # For LLM calls specifically
  retry_delay: 1.0                      # seconds
  rate_limit_rpm: 60                    # requests / minute

  # Prompt-specific settings (part of LLMConfiguration)
  default_temperature: 0.2              # Unless prompt specifies otherwise
  default_max_tokens: 2048              # Default max tokens for LLM responses

  # Cost management
  max_tokens_per_request: 4000          # Max tokens for a single LLM request
  enable_cost_tracking: true
  monthly_budget_limit: null            # USD – triggers hard stop if exceeded

  # Note: 'strict_json_validation' mentioned in older guides is not a direct
  # YAML key in LLMConfiguration. JSON validation is typically handled by the
  # LLM provider or parsing logic within agents.
```

All provider-specific environment variables (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `AZURE_API_BASE`, `HF_TOKEN`) are supported via LiteLLM
and map to `llm.api_key` or are used directly by LiteLLM.

---

## 3. Logging (`logging`)

Configuration for system logging.

```yaml
logging:
  level: INFO                           # DEBUG | INFO | WARNING | ERROR
  enable_file_logging: true             # Enable logging to files
  log_directory: ./logs                 # Log file directory (if file logging enabled)
  # json_format in the guide previously mapped to enable_structured_logging
  enable_structured_logging: false      # Use structured JSON logging

  # Other settings from config_manager.LoggingConfiguration include:
  # include_timestamps, include_caller_info, max_log_size_mb,
  # log_retention_days, compress_old_logs, mask_secrets,
  # log_api_requests, enable_debug_logging
```

### Environment Variables
| Var                     | Maps to                       | Example                               |
|-------------------------|-------------------------------|---------------------------------------|
| `CHUNGOID_LOG_LEVEL`    | `logging.level`               | `export CHUNGOID_LOG_LEVEL=DEBUG`     |
| `CHUNGOID_ENABLE_DEBUG` | `logging.enable_debug_logging`| `export CHUNGOID_ENABLE_DEBUG=true`   |

---

## 4. Orchestrator Settings (Conceptual)

The `config_guide.md` previously listed an `orchestrator:` block with keys like
`shared_context_cache_size` and `enable_metrics`.

-   **`shared_context_cache_size`**: This is not found as a direct YAML
    configuration key in `config_manager.py` for managing the size of stage
    outputs in the shared context. Caching strategies might be implemented
    internally (e.g., `AgentResolutionCache` for agent discovery).
-   **`enable_metrics`**: The closest equivalent is `agents.enable_performance_monitoring`
    (see Section 1), which controls broader agent performance tracking.

Specific orchestrator behaviors are generally influenced by settings within the
`agents:` configuration block (like `max_concurrent_agents`, checkpointing settings)
or are intrinsic to the `UnifiedOrchestrator`'s logic.

---

## 5. Example `config.yaml` Profiles

Below are three ready-made profiles illustrating how you might tune Chungoid.
These have been updated to reflect the current configuration structure.

### 5.1 Low-Complexity / Fast
Focuses on speed for very small tasks or demos.
```yaml
llm:
  provider: openai
  default_model: gpt-3.5-turbo
  timeout: 30
  default_temperature: 0.5 # Moved from prompt block

agents:
  default_max_iterations: 2         # Quick double-check pass
  # execution_mode, min_quality_score are conceptual, not direct YAML under agents

logging:
  level: INFO
```

### 5.2 Medium-Complexity / Balanced
Suitable for typical backend or web-app sized projects.
```yaml
llm:
  provider: openai
  default_model: gpt-4o-mini-2024-07-18
  fallback_model: gpt-3.5-turbo
  timeout: 60
  max_retries: 4 # LLM specific retries
  default_temperature: 0.3 # Moved from prompt block

agents:
  default_max_iterations: 6
  # execution_mode, min_quality_score are conceptual
  max_retries: 3 # Agent/stage retries
  enable_performance_monitoring: true # Was orchestrator.enable_metrics

logging:
  level: DEBUG
```

### 5.3 High-Complexity / Maximum
Designed for very large mono-repos or multi-language systems.
```yaml
llm:
  provider: openai
  default_model: gpt-4o
  timeout: 120
  max_retries: 6 # LLM specific retries
  rate_limit_rpm: 180
  enable_cost_tracking: true
  monthly_budget_limit: 500.0      # USD
  default_temperature: 0.2 # Moved from prompt block
  # strict_json_validation is not a direct YAML config

agents:
  default_max_iterations: 20       # Generous budget
  # execution_mode, min_quality_score, stop_on_success are conceptual
  enable_performance_monitoring: true # Was orchestrator.enable_metrics
  # Note: shared_context_cache_size is not a direct YAML config

logging:
  level: DEBUG
  enable_structured_logging: true # Was json_format
  log_directory: ./logs # Ensure path consistency, file name is auto-generated
```

---

## 6. Provider setup quick-reference
(This section remains largely the same as it refers to external env vars)

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Ollama (local)
Example for your `config.yaml`:
```yaml
llm:
  provider: ollama
  default_model: mistral # Or your preferred Ollama model
  api_base_url: http://localhost:11434
```

Et cetera – see LiteLLM docs for more providers.

---

## 7. Viewing the effective configuration
```bash
chungoid utils show-config            # merged view
chungoid utils show-config --raw      # raw YAML
```
(Assuming these CLI commands are still current)

---

*Last major revision based on `config_manager.py` structure.* 