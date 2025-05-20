"""Utility function for loading JSON schema files."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

class JSONSchemaLoadError(Exception):
    """Custom exception for errors during JSON schema loading."""
    pass

def load_json_schema_from_file(schema_file_path: Path) -> Dict[str, Any]:
    """
    Loads a JSON schema from the specified file path.

    Args:
        schema_file_path: The Path object pointing to the JSON schema file.

    Returns:
        A dictionary representing the loaded JSON schema.

    Raises:
        JSONSchemaLoadError: If the file cannot be found, read, or parsed as JSON.
    """
    logger.debug(f"Attempting to load JSON schema from: {schema_file_path}")
    if not schema_file_path.is_file():
        err_msg = f"JSON schema file not found: {schema_file_path}"
        logger.error(err_msg)
        raise JSONSchemaLoadError(err_msg)

    try:
        with open(schema_file_path, 'r', encoding='utf-8') as f:
            schema_data = json.load(f)
        logger.info(f"Successfully loaded JSON schema from: {schema_file_path}")
        return schema_data
    except json.JSONDecodeError as e:
        err_msg = f"Error decoding JSON from schema file '{schema_file_path}': {e}"
        logger.error(err_msg, exc_info=True)
        raise JSONSchemaLoadError(err_msg) from e
    except IOError as e:
        err_msg = f"Error reading JSON schema file '{schema_file_path}': {e}"
        logger.error(err_msg, exc_info=True)
        raise JSONSchemaLoadError(err_msg) from e
    except Exception as e:
        err_msg = f"An unexpected error occurred while loading JSON schema from '{schema_file_path}': {e}"
        logger.error(err_msg, exc_info=True)
        raise JSONSchemaLoadError(err_msg) from e

__all__ = ["load_json_schema_from_file", "JSONSchemaLoadError"] 