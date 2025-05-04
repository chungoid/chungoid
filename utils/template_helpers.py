"""Helper functions for manipulating template strings or structures."""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)


def inject_variable(template: str, variable_name: str, value: str) -> str:
    """Injects a simple string variable into a template.

    Replaces placeholders like {{variable_name}}.

    Args:
        template: The template string.
        variable_name: The name of the variable (without brackets).
        value: The string value to inject.

    Returns:
        The template with the variable injected.
    """
    placeholder = f"{{{{{variable_name}}}}}"
    logger.debug("Injecting variable '%s' into template.", variable_name)
    return template.replace(placeholder, value)


def inject_multiple_variables(template: str, variables: Dict[str, str]) -> str:
    """Injects multiple string variables into a template.

    Args:
        template: The template string.
        variables: A dictionary where keys are variable names and values are strings.

    Returns:
        The template with all variables injected.
    """
    logger.debug("Injecting multiple variables: %s", list(variables.keys()))
    modified_template = template
    for name, value in variables.items():
        modified_template = inject_variable(modified_template, name, value)
    return modified_template


def format_list_section(
    template: str, section_name: str, items: List[str], item_format: str = "- {item}"
) -> str:
    """Formats a list of items into a specific section of a template.

    Replaces a placeholder like ### SECTION_NAME ###\n...content...\n### END_SECTION_NAME ###
    or simply {{section_name}} with a formatted list.

    Args:
        template: The main template string.
        section_name: The name of the section (used in placeholders).
        items: A list of strings to format.
        item_format: A format string for each item, where {item} is the placeholder
                     for the list item itself (default: "- {item}").

    Returns:
        The template with the list formatted into the specified section.
    """
    formatted_list = "\n".join(item_format.format(item=item) for item in items)
    logger.debug("Formatting list section '%s' with %d items.", section_name, len(items))

    # Try replacing block placeholder first
    block_placeholder_start = f"### {section_name.upper()} ###"
    block_placeholder_end = f"### END_{section_name.upper()} ###"
    pattern = re.compile(
        rf"{re.escape(block_placeholder_start)}.*?{re.escape(block_placeholder_end)}",
        re.DOTALL,
    )

    if pattern.search(template):
        logger.debug("Found block placeholder for section '%s'.", section_name)
        # Ensure newline before and after the list if replacing a block
        replacement_text = f"{block_placeholder_start}\n{formatted_list}\n{block_placeholder_end}"
        return pattern.sub(replacement_text, template)
    else:
        # Fallback to simple variable placeholder
        logger.debug(
            "Block placeholder not found, using simple placeholder for section '%s'.",
            section_name,
        )
        return inject_variable(template, section_name, formatted_list)


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    base_template = """
    Project: {{project_name}}
    Version: {{version}}

    ### FEATURES ###
    Placeholder for feature list.
    Will be replaced.
    ### END_FEATURES ###

    Dependencies:
    {{dependencies}}
    """

    variables = {"project_name": "Chungoid MCP", "version": "1.0.0"}

    features = ["Stage Execution", "Status Management", "Prompt Loading"]

    deps = ["fastmcp", "python-dotenv", "filelock"]

    # Inject simple variables
    temp_template = inject_multiple_variables(base_template, variables)
    logger.info("Template after variable injection:\n%s", temp_template)

    # Format feature list into its block section
    temp_template = format_list_section(temp_template, "FEATURES", features)
    logger.info("Template after feature list injection:\n%s", temp_template)

    # Format dependencies list using simple placeholder
    final_template = format_list_section(
        temp_template, "dependencies", deps, item_format="* {item}"
    )
    logger.info("Final Template:\n%s", final_template)
