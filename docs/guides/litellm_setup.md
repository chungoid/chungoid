# LiteLLM Setup Guide for Chungoid

This guide explains how to configure LiteLLM for use within the Chungoid framework. LiteLLM allows Chungoid to interact with a wide variety of Large Language Models (LLMs) from different providers using a unified interface.

## Overview

Chungoid's `LLMManager` uses `LiteLLMProvider` to make calls to LLMs. LiteLLM itself handles the specifics of connecting to different model providers (OpenAI, Anthropic, Ollama, Azure, HuggingFace, etc.).

The primary configuration for `LLMManager` (and thus LiteLLM) happens in your project's `project_config.yaml` or the global `config.yaml`.

## Configuration Values

Here are the key configuration values under the `llm_manager` section in your `config.yaml` that affect LiteLLM:

```yaml
llm_manager:
  provider_type: "litellm"  # Specifies that LiteLLM should be used.
  default_model: "ollama/mistral" # IMPORTANT: Set your desired default model.
                            # Format depends on the provider, e.g.:
                            # - OpenAI: "gpt-4-turbo-preview", "gpt-3.5-turbo"
                            # - Anthropic: "claude-3-opus-20240229", "claude-2.1"
                            # - Ollama: "ollama/mistral", "ollama/llama2", "ollama/codellama"
                            #   (prefix with "ollama/" for local Ollama models)
                            # - Azure: "azure/<your-deployment-name>"
                            # - HuggingFace: "huggingface/<model-repo-id>"
                            # Refer to LiteLLM documentation for more model strings.

  api_key: null             # Optional: Directly provide an API key.
                            # LiteLLM typically prefers API keys to be set as environment variables
                            # (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY).
                            # If set here, it can be used as a fallback or for specific providers
                            # that LiteLLM can use it for. `null` means rely on environment variables.

  base_url: null            # Optional: For self-hosted models or custom API endpoints.
                            # - For local Ollama: "http://localhost:11434" (or your Ollama server URL)
                            # - For Azure OpenAI: Your Azure endpoint.
                            # - For other OpenAI-compatible servers.

  provider_env_vars: {}     # Optional: A dictionary of environment variables to set programmatically
                            # specifically for LiteLLM's context when it initializes.
                            # Example:
                            # provider_env_vars:
                            #   OPENAI_API_KEY: "sk-yourkey" # Not recommended for sensitive keys
                            #   AZURE_API_VERSION: "2023-07-01-preview"
                            #   HF_TOKEN: "your_hf_token_for_private_models"

  # MockLLMProvider specific (if provider_type is "mock"):
  mock_llm_responses:
    "initial prompt part": "mocked response for this prompt"
    # ... more mock responses
```

## Setting Up Specific LLM Providers via LiteLLM

LiteLLM attempts to automatically detect provider credentials from standard environment variables.

### 1. OpenAI Models (e.g., GPT-4, GPT-3.5)

-   **Environment Variable**: Set `OPENAI_API_KEY` to your OpenAI API key.
    ```bash
    export OPENAI_API_KEY="sk-yourActualOpenAIKey"
    ```
-   **Config Example (`default_model`)**:
    ```yaml
    llm_manager:
      default_model: "gpt-4-turbo-preview"
    ```

### 2. Anthropic Models (e.g., Claude 3)

-   **Environment Variable**: Set `ANTHROPIC_API_KEY` to your Anthropic API key.
    ```bash
    export ANTHROPIC_API_KEY="sk-ant-yourActualAnthropicKey"
    ```
-   **Config Example (`default_model`)**:
    ```yaml
    llm_manager:
      default_model: "claude-3-opus-20240229"
    ```

### 3. Ollama (Local LLMs)

Ensure your Ollama server is running (typically at `http://localhost:11434`).

-   **No API Key Needed** for local Ollama by default.
-   **Config Example**:
    ```yaml
    llm_manager:
      default_model: "ollama/mistral" # Or "ollama/llama2", "ollama/codellama", etc.
      base_url: "http://localhost:11434" # Or your Ollama server address
    ```
    *Note: The `ollama/` prefix in the `default_model` tells LiteLLM to treat it as an Ollama model.*

### 4. Azure OpenAI Service

-   **Environment Variables**:
    ```bash
    export AZURE_API_KEY="yourAzureOpenAIKey"
    export AZURE_API_BASE="https://your-resource-name.openai.azure.com/" # Your Azure endpoint
    export AZURE_API_VERSION="2023-07-01-preview" # Or your preferred API version
    ```
    Alternatively, you can set these in `provider_env_vars` or `api_key` / `base_url` in the config.
-   **Config Example (`default_model`)**:
    ```yaml
    llm_manager:
      default_model: "azure/your-azure-deployment-name" # Replace with your actual deployment name
      # api_key: "yourAzureOpenAIKey" # Can also be set here
      # base_url: "https://your-resource-name.openai.azure.com/" # Can also be set here
      # provider_env_vars:
      #   AZURE_API_VERSION: "2023-07-01-preview"
    ```

### 5. HuggingFace Inference Endpoints (and other compatible servers)

-   **Environment Variable (for gated/private models)**: `HF_TOKEN`
    ```bash
    export HF_TOKEN="yourHuggingFaceToken"
    ```
-   **Config Example**:
    ```yaml
    llm_manager:
      default_model: "huggingface/mistralai/Mistral-7B-Instruct-v0.1" # Public model example
      # For a model on an inference endpoint:
      # default_model: "huggingface/your-repo-id"
      # base_url: "https://your-inference-endpoint-url" # If using a dedicated HF endpoint
    ```

### 6. Other Providers

LiteLLM supports many other providers (Google Vertex AI/Gemini, Cohere, Bedrock, etc.).
-   Generally, set the provider-specific API key environment variable (e.g., `GOOGLE_API_KEY`).
-   Set the `default_model` to the LiteLLM string for that model.
-   Refer to the [LiteLLM Documentation](https://docs.litellm.ai/docs/providers) for the correct model strings and required environment variables for each provider.

## Per-Prompt Model Configuration

While `llm_manager.default_model` sets the system-wide default, individual prompts defined in YAML files (e.g., in `server_prompts/`) can also specify their own model:

```yaml
# Example: server_prompts/autonomous_engine/some_agent_prompt.yaml
id: "some_agent_prompt_v1"
version: "1.0"
# ... other metadata ...
model_config:
  model_id: "ollama/codellama:13b" # This prompt will use codellama via Ollama
  temperature: 0.5
  max_tokens: 3000
  # response_format: { "type": "json_object" } # If JSON output is expected
# ... prompt templates ...
```
If a prompt definition includes a `model_id`, it will override the `default_model` from `config.yaml` for calls made using that specific prompt.

## Important Notes

-   **LiteLLM Installation**: Ensure LiteLLM is installed in your Python environment. It should be part of `chungoid-core`'s dependencies. If not, `pip install litellm`.
-   **Environment Variables**: Setting API keys as environment variables is generally more secure than hardcoding them in config files, especially if your `config.yaml` is committed to version control. Use a `.env` file (which `LLMManager` loads via `python-dotenv`) or export them in your shell.
-   **Chungoid Config Loading**: Chungoid loads `config.yaml` from `chungoid-core/src/chungoid/config.yaml` by default. It can also load project-specific configurations from `<your_project_dir>/.chungoid/project_config.yaml`.
-   **Debugging LiteLLM**: If you encounter issues, LiteLLM has a verbose mode that can be helpful. This is usually enabled by `litellm.set_verbose = True` in code, which `LiteLLMProvider` might do or can be added for debugging. Check Chungoid's logging output for messages from `LiteLLMProvider` and LiteLLM itself.

This guide should help you configure various LLMs for use with Chungoid through LiteLLM. Always refer to the official [LiteLLM Documentation](https://docs.litellm.ai/) for the most up-to-date information on providers and model strings. 