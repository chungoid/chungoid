# LiteLLM Setup Guide for Chungoid

This guide explains how to configure LiteLLM for use within the Chungoid framework. LiteLLM allows Chungoid to interact with a wide variety of Large Language Models (LLMs) from different providers using a unified interface.

## Overview

Chungoid's `LLMManager` uses `LiteLLMProvider` to make calls to LLMs. LiteLLM itself handles the specifics of connecting to different model providers (OpenAI, Anthropic, Ollama, Azure, HuggingFace, etc.).

The primary configuration for `LLMManager` (and thus LiteLLM) happens in your project's `.chungoid/config.yaml` file using the new **Pydantic-based configuration system**.

## Configuration Values

Here are the key configuration values under the `llm` section in your `config.yaml` that affect LiteLLM:

```yaml
llm:
  provider: "openai"  # Specifies the LLM provider: "openai", "anthropic", "ollama", "litellm", or "mock"
  default_model: "gpt-4o-mini-2024-07-18" # IMPORTANT: Set your desired default model.
                            # Format depends on the provider, e.g.:
                            # - OpenAI: "gpt-4o", "gpt-4o-mini-2024-07-18", "gpt-3.5-turbo"
                            # - Anthropic: "claude-3-5-sonnet-20241022", "claude-3-opus-20240229"
                            # - Ollama: "mistral", "llama2", "codellama" (no prefix needed)
                            # - Azure: "azure/<your-deployment-name>"
                            # - HuggingFace: "huggingface/<model-repo-id>"
                            # Refer to LiteLLM documentation for more model strings.

  api_key: null             # Optional: Directly provide an API key (as SecretStr).
                            # LiteLLM typically prefers API keys to be set as environment variables
                            # (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
                            # If set here, it can be used as a fallback or for specific providers.
                            # `null` means rely on environment variables.

  api_base_url: null        # Optional: For self-hosted models or custom API endpoints.
                            # - For local Ollama: "http://localhost:11434" (or your Ollama server URL)
                            # - For Azure OpenAI: Your Azure endpoint.
                            # - For other OpenAI-compatible servers.

  fallback_model: "gpt-3.5-turbo"    # Fallback model if the default fails
  timeout: 60                        # Request timeout in seconds
  max_retries: 3                     # Maximum retry attempts
  retry_delay: 1.0                   # Delay between retries
  rate_limit_rpm: 60                 # Rate limit requests per minute
  max_tokens_per_request: 4000       # Maximum tokens per request
  enable_cost_tracking: true         # Enable cost tracking features
  monthly_budget_limit: null         # Optional monthly budget limit

  # Note: provider_env_vars is no longer directly supported in the new schema.
  # Use environment variables directly or configure them in your deployment environment.

  # MockLLMProvider specific (if provider is "mock"):
  # Mock responses are now handled differently in the new system.
  # See Mock LLM Provider section below for details.
```

## Setting Up Specific LLM Providers via LiteLLM

LiteLLM attempts to automatically detect provider credentials from standard environment variables.

### 1. OpenAI Models (e.g., GPT-4, GPT-3.5)

-   **Environment Variable**: Set `OPENAI_API_KEY` to your OpenAI API key.
    ```bash
    export OPENAI_API_KEY="sk-yourActualOpenAIKey"
    ```
-   **Config Example**:
    ```yaml
    llm:
      provider: "openai"
      default_model: "gpt-4o-mini-2024-07-18"
    ```

### 2. Anthropic Models (e.g., Claude 3)

-   **Environment Variable**: Set `ANTHROPIC_API_KEY` to your Anthropic API key.
    ```bash
    export ANTHROPIC_API_KEY="sk-ant-yourActualAnthropicKey"
    ```
-   **Config Example**:
    ```yaml
    llm:
      provider: "anthropic"
      default_model: "claude-3-5-sonnet-20241022"
    ```

### 3. Ollama (Local LLMs)

Ensure your Ollama server is running (typically at `http://localhost:11434`).

-   **No API Key Needed** for local Ollama by default.
-   **Config Example**:
    ```yaml
    llm:
      provider: "ollama"
      default_model: "mistral" # No prefix needed with new system
      api_base_url: "http://localhost:11434" # Or your Ollama server address
    ```

### 4. Azure OpenAI Service

-   **Environment Variables**:
    ```bash
    export AZURE_API_KEY="yourAzureOpenAIKey"
    export AZURE_API_BASE="https://your-resource-name.openai.azure.com/"
    export AZURE_API_VERSION="2023-07-01-preview"
    ```
-   **Config Example**:
    ```yaml
    llm:
      provider: "openai"  # Use openai provider for Azure
      default_model: "azure/your-azure-deployment-name"
      api_base_url: "https://your-resource-name.openai.azure.com/"
    ```

### 5. HuggingFace Inference Endpoints

-   **Environment Variable (for gated/private models)**: `HF_TOKEN`
    ```bash
    export HF_TOKEN="yourHuggingFaceToken"
    ```
-   **Config Example**:
    ```yaml
    llm:
      provider: "litellm"  # Use litellm provider for HuggingFace
      default_model: "huggingface/mistralai/Mistral-7B-Instruct-v0.1"
      # api_base_url: "https://your-inference-endpoint-url" # If using dedicated endpoint
    ```

### 6. Other Providers

LiteLLM supports many other providers (Google Vertex AI/Gemini, Cohere, Bedrock, etc.).
-   Generally, set the provider-specific API key environment variable (e.g., `GOOGLE_API_KEY`).
-   Set the `default_model` to the LiteLLM string for that model.
-   Use `provider: "litellm"` for providers not directly supported by the configuration system.
-   Refer to the [LiteLLM Documentation](https://docs.litellm.ai/docs/providers) for the correct model strings and required environment variables for each provider.

## Mock LLM Provider

For testing and development, you can use the mock provider:

```yaml
llm:
  provider: "mock"
  default_model: "mock-model"
```

Or set the environment variable:
```bash
export CHUNGOID_LLM_PROVIDER="mock"
```

## Per-Prompt Model Configuration

While `llm.default_model` sets the system-wide default, individual prompts defined in YAML files (e.g., in `server_prompts/`) can also specify their own model:

```yaml
# Example: server_prompts/autonomous_engine/some_agent_prompt.yaml
id: "some_agent_prompt_v1"
version: "1.0"
# ... other metadata ...
model_config:
  model_id: "codellama" # This prompt will use codellama
  temperature: 0.5
  max_tokens: 3000
  # response_format: { "type": "json_object" } # If JSON output is expected
# ... prompt templates ...
```
If a prompt definition includes a `model_id`, it will override the `default_model` from `config.yaml` for calls made using that specific prompt.

## Configuration File Location

The new configuration system looks for configuration in the following order:

1. **Project-specific**: `<your_project_dir>/.chungoid/config.yaml`
2. **Global**: `chungoid-core/config.yaml`
3. **Environment variables**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
4. **Defaults**: Built-in Pydantic model defaults

## Important Notes

-   **New Configuration System**: Chungoid now uses a **Pydantic-based configuration system** with automatic validation and environment variable integration.
-   **LiteLLM Installation**: Ensure LiteLLM is installed in your Python environment. It should be part of `chungoid-core`'s dependencies.
-   **Environment Variables**: Setting API keys as environment variables is generally more secure than hardcoding them in config files. Use a `.env` file or export them in your shell.
-   **Configuration Validation**: The new system automatically validates configuration and provides clear error messages for invalid settings.
-   **Debugging LiteLLM**: Check Chungoid's logging output for messages from `LiteLLMProvider` and LiteLLM itself. Enable debug logging with `--log-level DEBUG`.

## Viewing Current Configuration

You can view your current configuration using the CLI:

```bash
# Show effective merged configuration
chungoid utils show-config

# Show raw configuration file content
chungoid utils show-config --raw
```

This guide should help you configure various LLMs for use with Chungoid through LiteLLM. Always refer to the official [LiteLLM Documentation](https://docs.litellm.ai/) for the most up-to-date information on providers and model strings. 