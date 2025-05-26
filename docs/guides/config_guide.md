# Chungoid Configuration Guide

This document describes **every** configurable option available to end-users via
`config.yaml` (project-local or global) and relevant environment variables.  It
supersedes the older *litellm_setup.md*.

---

## 1. Unified Agent Execution Interface (`uaei`)

Controls enhanced-cycle behaviour for all agents.

```yaml
uaei:
  # Hard cap across all stages unless overridden per ExecutionConfig
  default_max_iterations: 5             # (int) 1 == single-pass

  # Preferred mode when *execute()* is called without explicit mode
  execution_mode: optimal               # (enum) single_pass | multi_iteration | optimal

  # Quality gate that terminates early when reached (0-1)
  min_quality_score: 0.90               # (float)

  # Whether a SUCCESS completion_reason should end the loop immediately
  stop_on_success: true                 # (bool)

  # Default quality threshold for single-pass tasks
  quality_threshold: 0.85               # (float)
```

**Where it maps in code**  
`ExecutionConfig.quality_threshold`, `CompletionCriteria.min_quality_score`,
`default_max_iterations` → reflected into `ExecutionConfig.max_iterations` when
not supplied explicitly.

### Environment Variables
| Var | Maps to | Example |
|-----|---------|---------|
| `CHUNGOID_UAEI_MAX_ITER` | `default_max_iterations` | `export CHUNGOID_UAEI_MAX_ITER=10` |
| `CHUNGOID_UAEI_MODE` | `execution_mode` | `export CHUNGOID_UAEI_MODE=multi_iteration` |

---

## 2. LLM / LiteLLM Provider (`llm`)

(unchanged section refined and moved from previous guide)

```yaml
llm:
  provider: openai                      # openai | anthropic | ollama | litellm | mock
  default_model: gpt-4o-mini-2024-07-18 # String recognised by LiteLLM
  fallback_model: gpt-3.5-turbo         # Used if primary fails
  api_key: null                         # Prefer env vars e.g. OPENAI_API_KEY
  api_base_url: null                    # Custom endpoints / Ollama URL
  timeout: 60                           # seconds
  max_retries: 3
  retry_delay: 1.0                      # seconds
  rate_limit_rpm: 60                    # requests / minute
  max_tokens_per_request: 4000
  enable_cost_tracking: true
  monthly_budget_limit: null            # USD – triggers hard stop if exceeded
```

All provider-specific environment variables (e.g. `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `AZURE_API_BASE`, `HF_TOKEN`) are supported via LiteLLM.

---

## 3. Prompt & Template Settings (`prompt`)

```yaml
prompt:
  default_temperature: 0.2              # Unless prompt specifies otherwise
  default_max_tokens: 2048
  strict_json_validation: true          # Fail fast if model returns invalid JSON
```

---

## 4. Orchestrator (`orchestrator`)

```yaml
orchestrator:
  shared_context_cache_size: 100        # How many stage outputs to cache
  enable_metrics: true                  # Emit Prometheus metrics
```

---

## 5. Logging (`logging`)

```yaml
logging:
  level: info                           # debug | info | warning | error
  file: ./logs/chungoid.log
  json_format: false
```

---

## 6. Example `config.yaml` Profiles

Below are three ready-made profiles illustrating how you might tune Chungoid for
projects of differing complexity.

### 6.1  Low-Complexity / Fast ("config.yaml")

Focuses on speed for very small tasks or demos.

```yaml
llm:
  provider: openai
  default_model: gpt-3.5-turbo
  timeout: 30

uaei:
  execution_mode: multi_iteration   # still iterative but minimal loops
  default_max_iterations: 2         # quick double-check pass
  min_quality_score: 0.80           # accept good-enough answers

prompt:
  default_temperature: 0.5

logging:
  level: info
```

### 6.2  Medium-Complexity / Balanced ("config.yaml")

Suitable for typical backend or web-app sized projects.

```yaml
llm:
  provider: openai
  default_model: gpt-4o-mini-2024-07-18
  fallback_model: gpt-3.5-turbo
  timeout: 60
  max_retries: 4

uaei:
  execution_mode: optimal          # agent decides but iteration budget allows depth
  default_max_iterations: 6
  min_quality_score: 0.90

orchestrator:
  shared_context_cache_size: 200

prompt:
  default_temperature: 0.3

logging:
  level: debug
```

### 6.3  High-Complexity / Maximum ("config.yaml")

Designed for very large mono-repos or multi-language systems that require
exhaustive iterative refinement.

```yaml
llm:
  provider: openai
  default_model: gpt-4o
  timeout: 120
  max_retries: 6
  rate_limit_rpm: 180
  enable_cost_tracking: true
  monthly_budget_limit: 500.0      # USD

uaei:
  execution_mode: multi_iteration  # force loops
  default_max_iterations: 20       # generous budget
  min_quality_score: 0.97          # near-perfect output required
  stop_on_success: false           # run full allocation even after first success

prompt:
  default_temperature: 0.2
  strict_json_validation: true

orchestrator:
  shared_context_cache_size: 1000
  enable_metrics: true

logging:
  level: debug
  json_format: true
  file: ./logs/chungoid_max.log
```

---

## Provider setup quick-reference

### OpenAI
```bash
export OPENAI_API_KEY="sk-…"
```

### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-…"
```

### Ollama (local)
```yaml
llm:
  provider: ollama
  default_model: mistral
  api_base_url: http://localhost:11434
```

Et cetera – see LiteLLM docs for more providers.

---

### Viewing the effective configuration
```bash
chungoid utils show-config            # merged view
chungoid utils show-config --raw      # raw YAML
```

---

*Last updated: 2025-05-26 — UAEI Phase-3 defaults switched to multi-iteration.* 