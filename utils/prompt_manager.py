"""Manages loading of core and stage-specific prompt files."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import yaml
from jinja2 import Environment, TemplateError, select_autoescape


# Custom Exception for Prompt Loading/Rendering Errors
class PromptLoadError(Exception):
    """Custom exception for errors during prompt loading or parsing."""

    pass  # No additional logic needed


class PromptRenderError(PromptLoadError):
    """Custom exception specifically for errors during Jinja2 template rendering."""

    pass


# --- Jinja Filters (Example - needs implementation if used) ---
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
    """Loads, caches, and renders prompts from individual YAML files using Jinja2."""

    def __init__(self, server_stages_dir: str, common_template_path: str):
        """Initializes the PromptManager.

        Args:
            server_stages_dir: Path to the directory containing stage-specific YAML template files (e.g., stage0.yaml) relative to the server installation.
            common_template_path: Path to the common YAML template file containing preamble/postamble.

        Raises:
            ValueError: If paths are empty.
            PromptLoadError: If directories/files don't exist or cannot be loaded.
        """
        self.logger = logging.getLogger(__name__)
        if not server_stages_dir:
            raise ValueError("Server stage template directory must be provided.")
        if not common_template_path:
            raise ValueError("Common template path must be provided.")

        # Resolve paths during initialization to ensure they are absolute and correct
        self.stages_dir = Path(server_stages_dir).resolve()
        if not self.stages_dir.is_dir():
            raise PromptLoadError(
                f"Resolved server stage template directory not found: {self.stages_dir}"
            )

        self.common_template_path = Path(common_template_path).resolve()
        if not self.common_template_path.is_file():
            raise PromptLoadError(
                f"Resolved common template file not found: {self.common_template_path}"
            )

        self.stage_definitions: Dict[Union[str, int, float], Dict[str, Any]] = {}
        self.common_template: Dict[str, str] = {}

        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=None,  # We load templates manually
            autoescape=select_autoescape(
                disabled_extensions=("txt", "yaml", "md")
            ),  # Disable autoescaping for prompt text
            trim_blocks=True,  # Helps control whitespace
            lstrip_blocks=True,
        )
        # Add custom filters
        self.jinja_env.filters["format_as_bullets"] = format_as_bullets
        self.jinja_env.filters["to_json"] = to_json_filter

        self.logger.info(
            "PromptManager initialized. Server Stages Dir: '%s', Common Template: '%s'",
            self.stages_dir,
            self.common_template_path,
        )
        self._load_common_template()  # Load common template first
        self._load_stage_definitions()  # Load and cache stage files

    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Loads and parses a single YAML file."""
        self.logger.debug("Loading YAML file: %s", file_path)
        if not file_path.is_file():
            raise PromptLoadError(f"YAML file not found: {file_path}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                raise PromptLoadError(f"YAML content is not a dictionary: {file_path}")
            self.logger.debug("Successfully loaded and parsed YAML: %s", file_path)
            return data
        except yaml.YAMLError as e:
            self.logger.error("Failed to parse YAML file '%s': %s", file_path, e)
            raise PromptLoadError(f"Invalid YAML format in {file_path}: {e}") from e
        except IOError as e:
            self.logger.error("Failed to read YAML file '%s': %s", file_path, e)
            raise PromptLoadError(f"Could not read YAML file {file_path}: {e}") from e
        except Exception as e:
            self.logger.exception("Unexpected error loading YAML file '%s': %s", file_path, e)
            raise PromptLoadError(f"Unexpected error loading {file_path}: {e}") from e

    def _extract_stage_number(self, filename: str) -> Optional[Union[int, float]]:
        """Extracts stage number (int or float) from filename like 'stage1.yaml' or 'stage0.5.yaml'."""
        if filename.startswith("stage") and filename.endswith(".yaml"):
            try:
                num_str = filename[len("stage") : -len(".yaml")]
                if "." in num_str:
                    return float(num_str)
                else:
                    return int(num_str)
            except ValueError:
                return None
        return None

    def _load_stage_definitions(self):
        """Loads stage definitions by scanning the stages directory for stage*.yaml files."""
        self.logger.info("Loading stage definitions from %s", self.stages_dir)
        self.stage_definitions = {}
        loaded_count = 0
        errors = [] # Collect errors

        for file_path in self.stages_dir.glob("stage*.yaml"):
            stage_number = self._extract_stage_number(file_path.name)
            if stage_number is not None:
                try:
                    stage_data = self._load_yaml_file(file_path)
                    # Basic validation of required keys
                    required_keys = ['system_prompt', 'user_prompt']
                    missing_keys = [k for k in required_keys if k not in stage_data or not isinstance(stage_data[k], str)]
                    if missing_keys:
                        raise PromptLoadError(f"Stage {stage_number} definition missing required string keys: {missing_keys}")

                    # Store the entire loaded data under the stage number key
                    self.stage_definitions[stage_number] = stage_data
                    loaded_count += 1
                    self.logger.debug(
                        "Loaded stage definition for %s from %s", stage_number, file_path.name
                    )
                    self.logger.info(f"Successfully loaded and stored stage definition for {stage_number} from {file_path.name}")
                except PromptLoadError as e:
                    self.logger.error(
                        "Failed to load stage definition from %s: %s", file_path.name, e
                    )
                    errors.append(f"{file_path.name}: {e}") # Add error details
                except Exception as e:  # Catch unexpected errors during loading
                    self.logger.exception(
                        "Unexpected error loading stage definition from %s: %s", file_path.name, e
                    )
                    errors.append(f"{file_path.name}: Unexpected error - {e}") # Add error details
            else:
                self.logger.warning("Skipping file with non-standard name: %s", file_path.name)

        if errors:
            error_details = "\n".join(errors)
            self.logger.error(
                f"Encountered {len(errors)} error(s) while loading stage definitions:\n{error_details}"
            )
            # Raise an error if any stage failed to load
            raise PromptLoadError(f"Failed to load {len(errors)} stage definition file(s). See logs for details.")

        if not self.stage_definitions:
            # If no errors occurred but still no definitions, it's a different issue
            self.logger.error(
                "No valid stage definition files (stage*.yaml) found in %s", self.stages_dir
            )
            raise PromptLoadError(f"No valid stage definition files found in {self.stages_dir}")

        self.logger.info(
            "Finished loading stage definitions. Found %d stages.", len(self.stage_definitions)
        )

    def get_stage_definition(self, stage_number: Union[int, float, str]) -> Dict[str, Any]:
        """Retrieves the definition for a specific stage number."""
        if not self.stage_definitions:
            self._load_stage_definitions()  # Ensure loaded (or attempt reload if empty)

        # Attempt lookup with the provided type first
        stage_def = self.stage_definitions.get(stage_number)

        # If not found, try converting type (float to int, or int to float if applicable)
        if not stage_def:
            self.logger.debug(
                "Stage %s not found with original type (%s). Trying type conversion.",
                stage_number,
                type(stage_number),
            )
            try:
                if isinstance(stage_number, float) and stage_number.is_integer():
                    stage_def = self.stage_definitions.get(int(stage_number))
                elif isinstance(stage_number, int):
                    stage_def = self.stage_definitions.get(float(stage_number))
            except Exception as e: # Catch potential errors during conversion/lookup
                 self.logger.warning(f"Error during type conversion lookup for stage {stage_number}: {e}")

        if not stage_def:
            self.logger.error("Definition for stage '%s' not found in loaded files.", stage_number)
            # Include available keys for debugging
            self.logger.debug(f"Available stage keys: {list(self.stage_definitions.keys())}")
            raise PromptLoadError(f"Stage {stage_number} definition not found.")
        return stage_def

    def _load_common_template(self):
        """Loads the common template YAML file."""
        self.logger.info("Loading common template from %s", self.common_template_path)
        try:
            common_data = self._load_yaml_file(self.common_template_path)
            # Expecting keys like 'preamble' and 'postamble'
            if not isinstance(common_data.get("preamble"), str) or not isinstance(
                common_data.get("postamble"), str
            ):
                raise PromptLoadError(
                    "Common template missing required 'preamble' or 'postamble' string keys."
                )
            self.common_template = {
                "preamble": common_data.get("preamble", ""),
                "postamble": common_data.get("postamble", ""),
            }
            self.logger.info("Successfully loaded common template.")
        except PromptLoadError as e:
            self.logger.error("Failed to load common template: %s", e)
            raise PromptLoadError(f"Could not load common template: {e}") from e
        except Exception as e:  # Catch unexpected errors
            self.logger.exception("Unexpected error loading common template: %s", e)
            raise PromptLoadError(f"Unexpected error loading common template: {e}") from e

    def get_rendered_prompt(
        self, stage_number: Union[int, float, str], context_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """Loads the specific stage YAML, merges data, and renders the full prompt using Jinja2.

        Args:
            stage_number: The number of the stage to render the prompt for.
            context_data: Additional context data to inject into the template (e.g., reflections_summary).

        Returns:
            The fully rendered prompt string for the specified stage.

        Raises:
            PromptLoadError: If stage definition cannot be loaded.
            PromptRenderError: If there is an error during template rendering.
        """
        self.logger.info("Generating rendered prompt for stage %s", stage_number)
        context_data = context_data or {}

        try:
            # Get stage definition (raises PromptLoadError if not found)
            stage_def = self.get_stage_definition(stage_number)

            # Safely get template strings
            system_prompt_template_str = stage_def.get("system_prompt", "")
            user_prompt_template_str = stage_def.get("user_prompt", "")
            common_preamble = self.common_template.get("preamble", "")
            common_postamble = self.common_template.get("postamble", "")

            if not system_prompt_template_str and not user_prompt_template_str:
                 raise PromptLoadError(f"Stage {stage_number} definition is missing both 'system_prompt' and 'user_prompt'.")

            # Merge context: Stage-specific context overrides common context if keys conflict
            # For now, just pass the provided context_data
            render_context = context_data

            # Render parts
            rendered_system = ""
            rendered_user = ""

            if system_prompt_template_str:
                system_template = self.jinja_env.from_string(system_prompt_template_str)
                rendered_system = system_template.render(render_context)

            if user_prompt_template_str:
                user_template = self.jinja_env.from_string(user_prompt_template_str)
                rendered_user = user_template.render(render_context)

            # Combine parts (ensure some spacing)
            full_prompt = f"{common_preamble}\n\n{rendered_system}\n\n{rendered_user}\n\n{common_postamble}".strip()

            self.logger.debug("Successfully rendered prompt for stage %s", stage_number)
            return full_prompt

        except PromptLoadError: # Re-raise errors from get_stage_definition
            raise
        except (TemplateError, TypeError, KeyError) as e:
            self.logger.error(
                "Failed to render prompt template for stage %s: %s", stage_number, e, exc_info=True # Log traceback
            )
            raise PromptRenderError(
                f"Error rendering prompt for stage {stage_number}: {type(e).__name__} - {e}"
            ) from e
        except Exception as e:
            self.logger.exception(
                "Unexpected error rendering prompt for stage %s: %s", stage_number, e
            )
            raise PromptRenderError(
                f"Unexpected error rendering prompt for stage {stage_number}: {e}"
            ) from e
