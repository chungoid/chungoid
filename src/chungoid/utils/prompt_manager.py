"""Manages loading of core and agent-specific prompt files."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import yaml
from jinja2 import Environment, TemplateError, select_autoescape
from pydantic import BaseModel, field_validator, Field


logger = logging.getLogger(__name__)

# Custom Exception for Prompt Loading/Rendering Errors
class PromptLoadError(Exception):
    """Custom exception for errors during prompt loading or parsing."""

    pass  # No additional logic needed


class PromptRenderError(PromptLoadError):
    """Custom exception specifically for errors during Jinja2 template rendering."""

    pass


# --- Pydantic Models for Prompt Structure ---
class PromptModelSettings(BaseModel):
    """Defines settings for the LLM model to be used with the prompt."""
    model_name: Optional[str] = None  # No default - use project config or LLMProvider default
    temperature: float = 0.7
    max_tokens: Optional[int] = 2048
    # Can add other OpenAI compatible settings like top_p, presence_penalty, etc.

class PromptDefinition(BaseModel):
    """Represents the structure of a loaded prompt YAML file."""
    id: str = Field(..., description="Unique identifier for the prompt (e.g., agent name or task type).")
    version: str = Field(..., description="Version of the prompt (e.g., v1, v1.1).")
    description: str = Field(..., description="A brief description of what the prompt is for.")
    system_prompt_template: str = Field(..., alias="system_prompt") # Allow alias for YAML key
    user_prompt_template: str = Field(..., alias="user_prompt")   # Allow alias for YAML key
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for expected input variables in prompt_render_data.")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for the expected LLM output structure.")
    model_settings: PromptModelSettings = Field(default_factory=PromptModelSettings)

    @field_validator('system_prompt_template', 'user_prompt_template', mode='before')
    @classmethod
    def ensure_string_template(cls, value):
        if not isinstance(value, str):
            raise ValueError("Prompt templates must be strings.")
        return value


# --- Jinja Filters (Example - needs implementation if used elsewhere) ---
def format_as_bullets(items: List[str]) -> str:
    """Formats a list of strings as a bulleted list."""
    if not items:
        return ""
    return "\n".join(f"• {item}" for item in items)


def to_json_filter(value) -> str:
    """Converts a Python object to a JSON string (simple version)."""
    try:
        import json

        # Ensure proper indentation for readability in prompts if needed
        return json.dumps(value, indent=2)
    except Exception:
        return str(value)  # Fallback


class PromptManager:
    """Loads, caches, and renders prompts from YAML files using Jinja2, structured by PromptDefinition."""

    def __init__(self, prompt_directory_paths: List[str]):
        """Initializes the PromptManager.

        Args:
            prompt_directory_paths: A list of paths to directories containing prompt YAML files.
                                     Prompts are organized by subdirectories (e.g., 'autonomous_engine').
        Raises:
            PromptLoadError: If directories don't exist or no prompts are loaded.
        """
        self.logger = logging.getLogger(__name__)
        self.prompt_definitions: Dict[str, PromptDefinition] = {} # Key: "sub_path/prompt_name/version"
        self.jinja_env = self._create_jinja_environment()

        if not prompt_directory_paths:
            raise PromptLoadError("At least one prompt directory path must be provided.")

        for dir_path_str in prompt_directory_paths:
            dir_path = Path(dir_path_str).resolve()
            if not dir_path.is_dir():
                self.logger.warning(f"Prompt directory not found: {dir_path}")
                continue
            self._load_prompts_from_directory(dir_path)

        if not self.prompt_definitions:
            self.logger.error("No prompt definitions were loaded. Check directory paths and prompt file structures.")
            # Depending on strictness, could raise PromptLoadError here
            # For now, allowing initialization but get_prompt_definition will fail.

        self.logger.info(f"PromptManager initialized. Loaded {len(self.prompt_definitions)} prompt definitions.")

    def _create_jinja_environment(self) -> Environment:
        """Creates a Jinja2 environment with custom filters."""
        env = Environment(
            loader=None, # Templates are loaded as strings directly
            autoescape=select_autoescape(['html', 'xml']), # Though likely not needed for LLM prompts
            undefined=jinja2.StrictUndefined, # Raise error on undefined variables
        )
        env.filters['format_as_bullets'] = format_as_bullets
        env.filters['to_json'] = to_json_filter
        return env

    def _load_prompts_from_directory(self, base_dir_path: Path):
        """Recursively loads all prompt YAML files from a base directory and its subdirectories."""
        self.logger.info(f"Scanning for prompt YAML files in: {base_dir_path}")
        for yaml_file_path in base_dir_path.rglob("*.yaml"): # rglob for recursive
            try:
                relative_path = yaml_file_path.relative_to(base_dir_path)
                # sub_path is the parent directory of the prompt file (e.g., autonomous_engine)
                # prompt_name is the filename without .yaml (e.g., requirements_tracer_agent_v1_prompt)
                
                sub_path_parts = list(relative_path.parent.parts)
                sub_path_str = "/".join(sub_path_parts) if sub_path_parts else ""
                
                prompt_file_name = yaml_file_path.stem 
                # Assuming filename itself contains name and version, e.g., "my_agent_prompt_v1"
                # Or YAML content has id and version. We rely on YAML content for id & version.

                self.logger.debug(f"Attempting to load prompt definition from: {yaml_file_path}")
                with open(yaml_file_path, 'r', encoding='utf-8') as f:
                    prompt_data = yaml.safe_load(f)
                
                if not isinstance(prompt_data, dict):
                    self.logger.warning(f"Skipping non-dictionary YAML file: {yaml_file_path}")
                    continue

                # Use Pydantic to parse and validate
                prompt_def = PromptDefinition(**prompt_data)
                
                # Construct a unique key. Ensure sub_path is handled correctly if it's empty.
                # Prompt ID should be unique (e.g., agent name). Version is separate.
                # Example key: "autonomous_engine/requirements_tracer_agent/v1"
                # The YAML 'id' field should be the core name, e.g. "requirements_tracer_agent"
                # The YAML 'version' field is the version, e.g. "v1"
                
                key_parts = []
                if sub_path_str:
                    key_parts.append(sub_path_str)
                key_parts.append(prompt_def.id)
                key_parts.append(prompt_def.version)
                cache_key = "/".join(key_parts)

                if cache_key in self.prompt_definitions:
                    self.logger.warning(f"Duplicate prompt definition found for key '{cache_key}' from file {yaml_file_path}. Overwriting.")
                
                self.prompt_definitions[cache_key] = prompt_def
                self.logger.info(f"Successfully loaded and cached prompt: '{cache_key}' from {yaml_file_path}")

            except FileNotFoundError:
                self.logger.error(f"Prompt file not found during scan: {yaml_file_path}") # Should not happen with glob
            except yaml.YAMLError as e:
                self.logger.error(f"Error parsing YAML file '{yaml_file_path}': {e}")
            except ValueError as e: # Catches Pydantic validation errors or others
                self.logger.error(f"Error validating prompt data from '{yaml_file_path}': {e}")
            except Exception as e:
                self.logger.error(f"Unexpected error loading prompt from '{yaml_file_path}': {e}", exc_info=True)


    def get_prompt_definition(
        self, prompt_name: str, prompt_version: str, sub_path: Optional[str] = None
    ) -> PromptDefinition:
        """
        Retrieves a parsed PromptDefinition.

        Args:
            prompt_name: The base name of the prompt (e.g., 'requirements_tracer_agent', from YAML 'id' field).
            prompt_version: The version of the prompt (e.g., 'v1', from YAML 'version' field).
            sub_path: Optional subdirectory where the prompt is located (e.g., 'autonomous_engine').

        Returns:
            The PromptDefinition object.

        Raises:
            PromptLoadError: If the prompt definition is not found.
        """
        key_parts = []
        if sub_path:
            key_parts.append(sub_path)
        key_parts.append(prompt_name)
        key_parts.append(prompt_version)
        cache_key = "/".join(key_parts)
        
        prompt_def = self.prompt_definitions.get(cache_key)
        if not prompt_def:
            self.logger.error(f"Prompt definition not found for key: '{cache_key}'. Available keys: {list(self.prompt_definitions.keys())}")
            raise PromptLoadError(f"Prompt '{prompt_name}' version '{prompt_version}' (sub-path: '{sub_path}') not found.")
        return prompt_def

    def get_rendered_prompt_template(
        self, template_string: str, context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Renders a given prompt template string with the provided context data.

        Args:
            template_string: The Jinja2 template string.
            context_data: Data to render the prompt template.

        Returns:
            The rendered prompt string.

        Raises:
            PromptRenderError: If there is an error during template rendering.
        """
        if context_data is None:
            context_data = {}
        try:
            template = self.jinja_env.from_string(template_string)
            rendered_prompt = template.render(context_data)
            return rendered_prompt
        except TemplateError as e:
            self.logger.error(f"Error rendering prompt template: {e}. Template (first 100 chars): '{template_string[:100]}...' Context: {context_data}")
            raise PromptRenderError(f"Error rendering prompt template: {e}") from e
        except Exception as e:
            self.logger.error(f"Unexpected error rendering template: {e}. Template: '{template_string[:100]}...'", exc_info=True)
            raise PromptRenderError(f"Unexpected error during template rendering: {e}") from e

    # --- Deprecated/Old methods that might need removal or adaptation ---
    # The following methods are from the older version of PromptManager seen in the snippet.
    # They need to be reviewed. get_rendered_prompt is similar to what an agent might call,
    # but it would now use get_prompt_definition first.

    # def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]: ... (covered by new loading logic)
    # def _extract_stage_number(self, filename: str) -> Optional[Union[int, float]]: ... (not relevant for new structure)
    # def _load_stage_definitions(self): ... (replaced by _load_prompts_from_directory)
    # def get_stage_definition(self, stage_number: Union[int, float, str]) -> Dict[str, Any]: ... (replaced by get_prompt_definition)
    # def _load_common_template(self): ... (common templates concept needs rethinking if still needed)
    
    # def get_rendered_prompt(
    #     self, stage_number: Union[int, float, str], context_data: Optional[Dict[str, Any]] = None
    # ) -> str:
    # Might be useful as a helper for agents if they don't want to call get_prompt_definition then render themselves.
    # Example of a convenience method:
    async def get_rendered_system_and_user_prompts(
        self, 
        prompt_name: str, 
        prompt_version: str, 
        prompt_render_data: Dict[str, Any],
        prompt_sub_path: Optional[str] = None
    ) -> (str, str):
        prompt_def = self.get_prompt_definition(prompt_name, prompt_version, sub_path=prompt_sub_path)
        system_prompt = self.get_rendered_prompt_template(prompt_def.system_prompt_template, prompt_render_data)
        user_prompt = self.get_rendered_prompt_template(prompt_def.user_prompt_template, prompt_render_data)
        return system_prompt, user_prompt

# Ensure jinja2 is imported if StrictUndefined is used directly
import jinja2

"""
Example YAML prompt file structure (e.g., chungoid-core/server_prompts/autonomous_engine/my_agent_v1_prompt.yaml):

id: "my_agent" # Base name of the agent/prompt
version: "v1"
description: "Prompt for My Agent to do X."
model_settings:
  model_name: "gpt-4o-mini-2024-07-18"
  temperature: 0.5
  max_tokens: 1500
system_prompt: |
  You are My Agent. Your goal is to {{ goal }}.
  Always respond in JSON.
user_prompt: |
  Based on the input: {{ input_data }}, perform your task.
  Details: {{ details }}
input_schema:
  type: "object"
  properties:
    goal: { type: "string" }
    input_data: { type: "string" }
    details: { type: "string" }
  required: ["goal", "input_data"]
output_schema:
  type: "object"
  properties:
    result: { type: "string" }
    confidence: { type: "number" }
  required: ["result", "confidence"]

"""
