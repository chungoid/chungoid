import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import re # For parsing more complex index/key access

from pydantic import BaseModel # ADDED
from chungoid.schemas.orchestration import SharedContext
from chungoid.schemas.project_status_schema import ArtifactDetails
# Placeholder for MasterStageSpec if specific input specs are handled here later
# from chungoid.schemas.master_flow import MasterStageSpec

logger = logging.getLogger(__name__)

class ContextResolutionService:
    """
    Handles the resolution of context paths and input values for stages
    within the orchestration flow.

    This service is responsible for interpreting various path expressions
    (e.g., '{context.outputs.some_stage.key}', '@artifacts.my_artifact')
    and retrieving the corresponding values from the SharedContext or
    other relevant sources.
    """

    # Regex to find {context...} paths and also general dot-separated paths for non-context.
    # It also captures list indices like [0] and quoted dict keys like ["my-key"] or ['my-key']
    _accessor_regex = re.compile(r"""
        \.([a-zA-Z0-9_-]+)               # Attribute access: .key_name (allows underscore, hyphen)
        |\[(\'[^\']+\')\]    # Quoted key access: [\'key_name\']
        |\[("[^"]+")\]  # Quoted key access: ["key_name"]
        |\[(\d+)\]             # List index access: [0]
    """, re.VERBOSE)

    def __init__(self, shared_context: Optional[SharedContext] = None, logger: Optional[logging.Logger] = None):
        """
        Initializes the ContextResolutionService.

        Args:
            shared_context: The shared context instance containing outputs,
                            artifacts, and other data accessible during the flow.
            logger: An optional logger instance. If not provided, a new one
                    will be created.
        """
        self.shared_context = shared_context
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("ContextResolutionService initialized.")

    def _resolve_value(self, value_spec: Any, current_shared_context: SharedContext) -> Any:
        """
        Resolves a single value, which might be a path string, a dict, a list, or a literal.
        """
        if isinstance(value_spec, str):
            if value_spec.startswith("@"):
                try:
                    return self._resolve_at_path(value_spec, current_shared_context)
                except Exception as e:
                    self.logger.warning(f"Failed to resolve @path '{value_spec}': {e}")
                    return None
            elif value_spec.startswith("{context.") and value_spec.endswith("}"):
                path_expression_in_braces = value_spec[len("{context."):-1]
                path_for_resolver: str
                if path_expression_in_braces.startswith("outputs."):
                    path_for_resolver = path_expression_in_braces
                elif path_expression_in_braces.startswith("prev_outputs."):
                    path_for_resolver = path_expression_in_braces
                else:
                    path_for_resolver = f"context.{path_expression_in_braces}"
                try:
                    return self.resolve_single_path(path_for_resolver, current_shared_context)
                except Exception as e:
                    self.logger.warning(f"Failed to resolve context path '{value_spec}' (resolved as '{path_for_resolver}'): {e}")
                    return None
            elif value_spec.startswith("{") and value_spec.endswith("}"):  # General {path.to.value}
                path_expression = value_spec[1:-1]
                try:
                    return self.resolve_single_path(path_expression, current_shared_context)
                except Exception as e:
                    self.logger.warning(f"Failed to resolve general path '{value_spec}': {e}")
                    return None
            else:
                return value_spec  # Literal string
        elif isinstance(value_spec, dict):
            # Recursively resolve dictionary values
            return {k: self._resolve_value(v, current_shared_context) for k, v in value_spec.items()}
        elif isinstance(value_spec, list):
            # Recursively resolve list items
            return [self._resolve_value(item, current_shared_context) for item in value_spec]
        else:
            return value_spec # Literal value (int, bool, etc.)

    def resolve_inputs_for_stage(
        self,
        inputs_spec: Dict[str, Any],
        # Allow overriding shared_context for more flexible use, e.g. in testing
        shared_context_override: Optional[SharedContext] = None
    ) -> Dict[str, Any]:
        """
        Resolves all specified input values for a stage based on their
        path expressions and the current shared context.
        Handles nested dictionaries and lists.
        """
        self.logger.debug(f"Resolving inputs for stage with spec: {inputs_spec}")
        resolved_inputs: Dict[str, Any] = {}
        current_shared_context = shared_context_override if shared_context_override else self.shared_context

        if not current_shared_context:
            self.logger.error("SharedContext is not available for input resolution.")
            return inputs_spec

        for key, value_spec in inputs_spec.items():
            resolved_inputs[key] = self._resolve_value(value_spec, current_shared_context)
        
        self.logger.info(f"Successfully resolved inputs (some may be None if resolution failed): {resolved_inputs}")
        return resolved_inputs

    def _resolve_at_path(self, path_expression: str, shared_context: SharedContext) -> Any:
        """
        Resolves paths starting with '@', e.g., "@artifact.my_artifact_id.content".
        """
        if not path_expression.startswith("@"):
            raise ValueError("Invalid @path format: Must start with '@'")

        parts = path_expression[1:].split('.', 2) # Split on first two dots: type, id, rest_of_path
        
        if len(parts) < 2: # Must have at least @type.id
            raise ValueError(f"Invalid @artifact path format: '{path_expression}'. Expected @type.id[.attribute...]")

        at_type = parts[0]
        item_id = parts[1]
        attribute_path_str = parts[2] if len(parts) > 2 else None

        if not item_id: # Added check for empty item_id
            raise ValueError(f"Invalid @artifact path format: '{path_expression}'. Item ID cannot be empty.")

        if at_type == "artifact":
            # Modern SharedContext stores project_status in data
            project_status = shared_context.data.get("project_status") if shared_context.data else None
            if not project_status or not hasattr(project_status, 'artifacts'):
                raise ValueError("project_status.artifacts not found in SharedContext.data")
            
            artifact = project_status.artifacts.get(item_id)
            if not artifact:
                raise KeyError(f"Artifact '{item_id}' not found in project_status.artifacts")

            if not attribute_path_str: # e.g. @artifact.my_id
                return artifact

            # Handle artifact attributes: .path, .metadata, .content
            if attribute_path_str == "path":
                return Path(artifact.path_on_disk) if artifact.path_on_disk else None
            elif attribute_path_str == "metadata":
                return artifact.metadata
            elif attribute_path_str.startswith("metadata."):
                metadata_key_path = attribute_path_str[len("metadata."):].split('.')
                return self._get_value_from_container(artifact.metadata, metadata_key_path)
            elif attribute_path_str == "content":
                if artifact.path_on_disk is None:
                    raise ValueError(f"Artifact '{item_id}' path_on_disk is None, cannot read content")
                
                file_path = Path(artifact.path_on_disk)
                if not file_path.is_absolute():
                    # Modern SharedContext stores project_root_path in data
                    project_root_path = shared_context.data.get("project_root_path") if shared_context.data else None
                    if project_root_path is None:
                        raise ValueError(
                            f"project_root_path must be set in SharedContext.data to resolve relative artifact content path: {file_path}"
                        )
                    file_path = Path(project_root_path) / file_path
                
                if not file_path.exists():
                    raise FileNotFoundError(f"Artifact content file not found at {file_path} for artifact '{item_id}'")
                
                # Assuming text content for now, could be extended for binary based on artifact.content_type
                with open(file_path, "r", encoding="utf-8") as f:
                    return f.read()
            else:
                raise ValueError(f"Unknown artifact attribute: {attribute_path_str} in path '{path_expression}'")
        else:
            raise ValueError(f"Unsupported @path type: '{at_type}' in path '{path_expression}'")

    def _get_value_from_container(self, container: Any, accessors: List[str]) -> Any:
        """
        Traverses a nested structure (dict, list, object) using a list of accessors.
        Supports list indices, dict keys (with quotes), and object attributes.
        """
        current = container
        
        # DEBUG: Add comprehensive logging
        self.logger.debug(f"_get_value_from_container called with container type: {type(container)}, accessors: {accessors}")
        if hasattr(container, '__dict__'):
            self.logger.debug(f"Container attributes: {list(container.__dict__.keys()) if container.__dict__ else 'No __dict__'}")
        if isinstance(container, BaseModel):
            self.logger.debug(f"Pydantic model fields: {list(container.model_fields.keys())}")
            self.logger.debug(f"Pydantic model values: {container.model_dump()}")
        
        for i, accessor in enumerate(accessors):
            self.logger.debug(f"Processing accessor[{i}]: '{accessor}' on current type: {type(current)}")
            
            # Check for list index notation like [0], [1], etc.
            if accessor.startswith('[') and accessor.endswith(']'):
                index_str = accessor[1:-1]
                try:
                    if index_str.startswith('"') and index_str.endswith('"'):
                        # Dict key with quotes like ["my-key"]
                        actual_accessor = index_str[1:-1]
                        self.logger.debug(f"Dict key access with quotes: '{actual_accessor}' on {type(current)}")
                        if isinstance(current, dict):
                            if actual_accessor in current:
                                current = current[actual_accessor]
                                self.logger.debug(f"Successfully accessed dict key '{actual_accessor}', result type: {type(current)}")
                            else:
                                self.logger.error(f"Dict key '{actual_accessor}' not found in dict with keys: {list(current.keys())}")
                                raise KeyError(f"Key '{actual_accessor}' not found in dict")
                        else:
                            self.logger.error(f"Attempted dict key access on non-dict type: {type(current)}")
                            raise TypeError(f"Cannot access key '{actual_accessor}' on non-dict object of type {type(current)}")
                    elif index_str.startswith("'") and index_str.endswith("'"):
                        # Dict key with single quotes like ['my-key']
                        actual_accessor = index_str[1:-1]
                        self.logger.debug(f"Dict key access with single quotes: '{actual_accessor}' on {type(current)}")
                        if isinstance(current, dict):
                            if actual_accessor in current:
                                current = current[actual_accessor]
                                self.logger.debug(f"Successfully accessed dict key '{actual_accessor}', result type: {type(current)}")
                            else:
                                self.logger.error(f"Dict key '{actual_accessor}' not found in dict with keys: {list(current.keys())}")
                                raise KeyError(f"Key '{actual_accessor}' not found in dict")
                        else:
                            self.logger.error(f"Attempted dict key access on non-dict type: {type(current)}")
                            raise TypeError(f"Cannot access key '{actual_accessor}' on non-dict object of type {type(current)}")
                    else:
                        # Numeric index for list
                        actual_accessor = int(index_str)
                        self.logger.debug(f"List index access: [{actual_accessor}] on {type(current)}")
                        if isinstance(current, (list, tuple)):
                            if 0 <= actual_accessor < len(current):
                                current = current[actual_accessor]
                                self.logger.debug(f"Successfully accessed list index [{actual_accessor}], result type: {type(current)}")
                            else:
                                self.logger.error(f"List index [{actual_accessor}] out of range for list of length {len(current)}")
                                raise IndexError(f"List index [{actual_accessor}] out of range")
                        else:
                            self.logger.error(f"Attempted list index access on non-list type: {type(current)}")
                            raise TypeError(f"Cannot access index [{actual_accessor}] on non-list object of type {type(current)}")
                except (ValueError, KeyError, IndexError, TypeError) as e:
                    self.logger.error(f"Error accessing container with accessor '[{index_str}]': {e}")
                    raise
            else:
                # Regular attribute/key access
                # Remove quotes if present for dict access
                if accessor.startswith('"') and accessor.endswith('"'):
                    actual_accessor = accessor[1:-1]
                elif accessor.startswith("'") and accessor.endswith("'"):
                    actual_accessor = accessor[1:-1]
                else:
                    actual_accessor = accessor
                
                self.logger.debug(f"Regular access: accessor='{accessor}', actual_accessor='{actual_accessor}' on {type(current)}")
                
                if isinstance(current, dict):
                    self.logger.debug(f"Dict access: looking for key '{actual_accessor}' in dict with keys: {list(current.keys())}")
                    if actual_accessor in current:
                        current = current[actual_accessor]
                        self.logger.debug(f"Successfully accessed dict key '{actual_accessor}', result type: {type(current)}")
                    else:
                        self.logger.error(f"Dict key '{actual_accessor}' not found in dict with keys: {list(current.keys())}")
                        raise KeyError(f"Key '{actual_accessor}' not found in dict")
                elif isinstance(current, (list, tuple)):
                    self.logger.debug(f"List/tuple access: looking for index '{actual_accessor}' in {type(current)} of length {len(current)}")
                    try:
                        index = int(actual_accessor)
                        if 0 <= index < len(current):
                            current = current[index]
                            self.logger.debug(f"Successfully accessed list index [{index}], result type: {type(current)}")
                        else:
                            self.logger.error(f"List index [{index}] out of range for list of length {len(current)}")
                            raise IndexError(f"List index [{index}] out of range")
                    except ValueError:
                        self.logger.error(f"Cannot convert '{actual_accessor}' to integer for list access")
                        raise ValueError(f"Cannot convert '{actual_accessor}' to integer for list access")
                elif isinstance(current, BaseModel): # Pydantic BaseModel check
                    self.logger.debug(f"Pydantic model access: looking for field '{actual_accessor}' in model {type(current)}")
                    self.logger.debug(f"Available Pydantic fields: {list(current.model_fields.keys())}")
                    if hasattr(current, actual_accessor):
                        current = getattr(current, actual_accessor)
                        self.logger.debug(f"Successfully accessed Pydantic field '{actual_accessor}', result type: {type(current)}, value: {current}")
                    else:
                        try:
                            current_dump = current.model_dump()
                            self.logger.debug(f"Fallback to model_dump(): {current_dump}")
                            if actual_accessor in current_dump:
                                current = current_dump[actual_accessor]
                                self.logger.debug(f"Successfully accessed via model_dump() key '{actual_accessor}', result type: {type(current)}, value: {current}")
                            else:
                                self.logger.error(f"Attribute/field '{actual_accessor}' not found on Pydantic model {type(current)} via direct access or in model_dump(). Available fields: {list(current.model_fields.keys())}, model_dump keys: {list(current_dump.keys())}")
                                raise AttributeError(f"Attribute/field '{actual_accessor}' not found on Pydantic model {type(current)}. Path part: {accessor}")
                        except Exception as e_pydantic_fallback:
                            self.logger.error(f"Error during Pydantic model_dump() fallback for '{actual_accessor}' on {type(current)}: {e_pydantic_fallback}. Path part: {accessor}")
                            raise AttributeError(f"Attribute/field '{actual_accessor}' not found on Pydantic model {type(current)} or fallback failed. Path part: {accessor}")
                else: # General object attribute access
                    self.logger.debug(f"General object access: looking for attribute '{accessor}' on {type(current)}")
                    if hasattr(current, '__dict__'):
                        self.logger.debug(f"Object attributes: {list(current.__dict__.keys()) if current.__dict__ else 'No __dict__'}")
                    # Use original accessor for getattr
                    if not hasattr(current, accessor):
                        self.logger.error(f"Object of type {type(current)} has no attribute '{accessor}'. Path part: {accessor}")
                        raise AttributeError(f"Object of type {type(current)} has no attribute '{accessor}'. Path part: {accessor}")
                    current = getattr(current, accessor)
                    self.logger.debug(f"Successfully accessed object attribute '{accessor}', result type: {type(current)}")
        
        self.logger.debug(f"Final resolution result: type={type(current)}, value={current}")
        return current

    def _resolve_path_value_from_base_and_parts(
        self,
        base_object_name: str,
        base_object: Any,
        parts: List[str],
        path_expression: str, # For logging
        shared_context_for_fallback: Optional[SharedContext] = None, # ADDED for fallback
        allow_partial: bool = False # ADDED for partial resolution
    ) -> Any:
        """
        Core private method to resolve path parts starting from a base object.
        """
        try:
            return self._get_value_from_container(base_object, parts)
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            # ENHANCED FALLBACK LOGIC
            if base_object_name == "context" and shared_context_for_fallback:
                # Handle common cases where paths like {context.data.project_id} are used
                # but the value is actually stored as a direct attribute
                if len(parts) == 2 and parts[0] == "data":
                    # Path like context.data.project_id - try context.project_id directly
                    field_name = parts[1]
                    if hasattr(shared_context_for_fallback, field_name):
                        self.logger.info(
                            f"Path '{path_expression}' (parsed as base='{base_object_name}', parts={parts}) resolution failed with {type(e).__name__}: {e}. "
                            f"Attempting fallback to direct attribute 'shared_context.{field_name}'."
                        )
                        return getattr(shared_context_for_fallback, field_name)
                
                # Legacy fallback logic for project_root_path
                is_context_global_config_project_dir = (
                    parts == ["global_config", "project_dir"] and
                    shared_context_for_fallback.data and
                    "project_root_path" in shared_context_for_fallback.data
                )
                is_context_project_root_path = (
                    parts == ["project_root_path"] and
                    shared_context_for_fallback.data and
                    "project_root_path" in shared_context_for_fallback.data
                )

                if is_context_global_config_project_dir:
                    self.logger.info(
                        f"Path '{path_expression}' (parsed as base='{base_object_name}', parts={parts}) resolution failed with {type(e).__name__}: {e}. "
                        f"Attempting fallback to 'shared_context.data.project_root_path'."
                    )
                    return shared_context_for_fallback.data.get("project_root_path")
                elif is_context_project_root_path:
                    self.logger.info(
                        f"Path '{path_expression}' (parsed as base='{base_object_name}', parts={parts}) resolution failed with {type(e).__name__}: {e}. "
                        f"Directly attempting 'shared_context.data.project_root_path' as it was requested."
                    )
                    return shared_context_for_fallback.data.get("project_root_path")

            self.logger.warning(f"Error resolving path part in '{path_expression}': {e}. Base object: {base_object_name}, Parts: {parts}")
            if allow_partial:
                return None
            raise # Re-raise the original error if no fallback handled it

    def _parse_accessors(self, path_str: str) -> List[str]:
        """
        Parses a dot-separated path string with potential indexing into a list of accessors.
        Example: "key1[0].key2['sub_key']" -> ["key1", "[0]", "key2", "['sub_key']"]
        Handles simple dot separation if regex doesn't match complex parts.
        """
        if not path_str:
            return []

        accessors: List[str] = []
        remaining_path = path_str

        # First, split by '.' to handle top-level attributes
        # Then, for each part, check if it contains '[]' for indexing/dict access
        # This is a simplified approach compared to a single complex regex for all cases.

        # Split by '.' that are NOT inside quotes within brackets
        # This is tricky. A simpler way is to split by '.' and then re-join parts that were inside brackets.
        # Or, use the _accessor_regex to find all individual components.

        # Let's use a method that iteratively consumes the path.
        # Start with the first part before any brackets or dots.
        match = re.match(r"([a-zA-Z0-9_-]+)(.*)", remaining_path)
        if match:
            accessors.append(match.group(1))
            remaining_path = match.group(2)
        
        # Now process the rest which should start with . or [
        while remaining_path:
            part_match = self._accessor_regex.match(remaining_path)
            if part_match:
                # Find which group matched to get the accessor
                # Group 1: .attribute
                # Group 2: ['key']
                # Group 3: [\"key\"]
                # Group 4: [index]
                if part_match.group(1): # .attribute
                    accessors.append(part_match.group(1))
                elif part_match.group(2): # ['key']
                    accessors.append(f"[{part_match.group(2)}]" if part_match.group(2) else "")
                elif part_match.group(3): # [\"key\"]
                    accessors.append(f"[{part_match.group(3)}]" if part_match.group(3) else "")
                elif part_match.group(4): # [index]
                    accessors.append(f"[{part_match.group(4)}]" if part_match.group(4) else "")
                else:
                    # Should not happen if regex is correct and one group matches
                    self.logger.error(f"Accessor regex matched but no group captured for remaining path: {remaining_path}")
                    break 
                remaining_path = remaining_path[part_match.end():]
            else:
                # If regex doesn't match, it means the path is malformed or ends unexpectedly
                if remaining_path: # If there's still unparsed path
                    self.logger.warning(f"Could not parse remaining accessor path: '{remaining_path}' from original '{path_str}'")
                break
        
        # self.logger.debug(f"Parsed '{path_str}' into accessors: {accessors}")
        return accessors

    def resolve_single_path(
        self,
        path_expression: str,
        shared_context: Optional[SharedContext] = None,
        allow_partial: bool = False 
    ) -> Any:
        """
        Resolves a complex path expression like "context.outputs.stage_name.key[0].attribute"
        or "file_inputs.my_file.content" or "@artifact.id.content".
        Restored original logic flow.
        """
        effective_shared_context = shared_context if shared_context else self.shared_context
        if not effective_shared_context:
            self.logger.error("SharedContext is not available for path resolution.")
            if allow_partial: return None
            raise ValueError("SharedContext is not available for path resolution.")

        original_path_expression_for_logging = path_expression # Keep for logging

        # Normalize: strip {} if present
        clean_path = path_expression
        if path_expression.startswith("{") and path_expression.endswith("}"):
            clean_path = path_expression[1:-1]
        
        self.logger.debug(f"Resolving path '{original_path_expression_for_logging}', clean_path='{clean_path}'")
        
        if not clean_path:
            self.logger.warning(f"Path expression '{original_path_expression_for_logging}' became empty after cleaning.")
            if allow_partial: return None
            # Raising an error for an empty path seems appropriate
            raise ValueError(f"Path expression '{original_path_expression_for_logging}' is empty or invalid.")

        if clean_path.startswith("@"):
            try:
                return self._resolve_at_path(clean_path, effective_shared_context)
            except Exception as e:
                self.logger.warning(f"Failed to resolve @path '{clean_path}': {e}")
                if allow_partial: return None
                raise 

        # Determine base object and parts to resolve
        # This logic mirrors the original structure more closely.
        path_parts_split = clean_path.split('.', 1)
        base_object_name = path_parts_split[0]
        remaining_path_str = path_parts_split[1] if len(path_parts_split) > 1 else None
        
        self.logger.debug(f"base_object_name='{base_object_name}', remaining_path_str='{remaining_path_str}'")
        
        base_object: Any = None

        if base_object_name == "context":
            if not remaining_path_str: # Path is just "{context}"
                return effective_shared_context.data if hasattr(effective_shared_context, 'data') else effective_shared_context
            
            # For "context.xxx", use the SharedContext object itself as the base
            # This allows resolving paths like "context.data.project_id", "context.project_id", etc.
            # The SharedContext object has both direct attributes (project_id, session_id, etc.)
            # and a data dict, so this approach handles both cases correctly
            base_object = effective_shared_context
            # `parts_to_resolve` will be parsed from `remaining_path_str`
            self.logger.debug(f"Using SharedContext as base_object for path starting with 'context'")

        elif base_object_name == "outputs":
            if not hasattr(effective_shared_context, 'data') or not isinstance(effective_shared_context.data.get("outputs"), dict):
                self.logger.warning(f"'data[\"outputs\"]' not found or not a dict on SharedContext for path: {original_path_expression_for_logging}")
                if allow_partial: return None
                raise AttributeError("'data[\"outputs\"]' not found or not a dict on SharedContext")
            base_object = effective_shared_context.data["outputs"]
            # `parts_to_resolve` will be parsed from `remaining_path_str`

        elif base_object_name == "prev_outputs": # Added this case based on _resolve_value
            if not hasattr(effective_shared_context, 'data') or not isinstance(effective_shared_context.data.get("prev_outputs"), dict):
                self.logger.warning(f"'data[\"prev_outputs\"]' not found or not a dict on SharedContext for path: {original_path_expression_for_logging}")
                if allow_partial: return None
                raise AttributeError("'data[\"prev_outputs\"]' not found or not a dict on SharedContext")
            base_object = effective_shared_context.data["prev_outputs"]
             # `parts_to_resolve` will be parsed from `remaining_path_str`

        else: # Assume other base_names are top-level keys in shared_context.data or direct attributes
            if hasattr(effective_shared_context, 'data') and isinstance(effective_shared_context.data, dict) and base_object_name in effective_shared_context.data:
                base_object = effective_shared_context.data[base_object_name]
                # `parts_to_resolve` will be parsed from `remaining_path_str`
            elif hasattr(effective_shared_context, base_object_name): # Direct attribute on SharedContext object
                base_object = effective_shared_context # The object itself is the base
                # `parts_to_resolve` will be parsed from `clean_path` directly by _parse_accessors
                # No, `base_object_name` is the first part, `remaining_path_str` is the rest.
                # So `_parse_accessors` should operate on `clean_path` if base_object is shared_context itself.
                # This needs care. If path is "run_id", base_object_name="run_id", remaining_path_str=None
                # The current split logic might not handle this perfectly if base_object_name is the *entire* path.

                # If remaining_path_str is None, it means the path was just the base_object_name
                if remaining_path_str is None:
                     return getattr(effective_shared_context, base_object_name) # Return the attribute directly
                # If there is a remaining_path_str, it means base_object_name was an attribute, and we need to delve deeper
                base_object = getattr(effective_shared_context, base_object_name)
                # `parts_to_resolve` will be parsed from `remaining_path_str`
            else:
                self.logger.warning(f"Base path element '{base_object_name}' not found in SharedContext or its data for path: {original_path_expression_for_logging}")
                if allow_partial: return None
                raise KeyError(f"Base path element '{base_object_name}' not found for '{original_path_expression_for_logging}'")

        if remaining_path_str is None:
            # Path was just a base name like "{context}", "{outputs}", or "{run_id}" (if run_id is direct attr)
            self.logger.debug(f"No remaining path, returning base_object: {type(base_object)}")
            return base_object # Return the base object itself

        parts_to_resolve = self._parse_accessors(remaining_path_str)
        self.logger.debug(f"parts_to_resolve={parts_to_resolve}")

        # Use the enhanced resolution method that includes fallback logic
        resolved_value = self._resolve_path_value_from_base_and_parts(
            base_object_name=base_object_name,
            base_object=base_object,
            parts=parts_to_resolve,
            path_expression=original_path_expression_for_logging,
            shared_context_for_fallback=effective_shared_context,
            allow_partial=allow_partial
        )
        
        self.logger.debug(f"Final resolved_value type={type(resolved_value)}, value={resolved_value}")
        return resolved_value

    def resolve_path_value_from_context(
        self,
        path_str: str,
        shared_context: SharedContext,
        current_stage_outputs: Optional[Dict[str, Any]] = None, # Not used by current resolve_single_path logic
        allow_partial: bool = False  # Added allow_partial
    ) -> Any:
        """
        Helper function to resolve a value from the shared context based on a path string.
        This is essentially a wrapper around resolve_single_path for more direct use when 
        the shared_context is explicitly passed.

        Args:
            path_str: The path string to resolve (e.g., "{context.outputs.stage_one.keyA}").
            shared_context: The SharedContext object to resolve against.
            current_stage_outputs: Deprecated/Not used by current resolution logic via resolve_single_path.
            allow_partial: If True, returns None on resolution failure instead of raising.

        Returns:
            The resolved value, or None if resolution fails and allow_partial is True.
        
        Raises:
            KeyError, AttributeError, IndexError, TypeError if resolution fails and allow_partial is False.
        """
        if not path_str:
            return None
        
        # Store the original shared_context if this service instance has one
        original_instance_sc = self.shared_context
        # Temporarily set the passed shared_context for this resolution call
        self.shared_context = shared_context 
        
        resolved_value = None
        try:
            # The `resolve_single_path` method will use `self.shared_context` which we just set.
            resolved_value = self.resolve_single_path(path_str, allow_partial=allow_partial) 
        finally:
            # Restore the original shared_context to avoid side effects on the service instance
            self.shared_context = original_instance_sc
            
        return resolved_value

    def _resolve_path_within_object(self, obj: Any, path_str: str, allow_partial: bool = False) -> Any:
        """
        Resolves a dot-separated path string (which can include list indices like [0]
        and quoted dict keys like ["my-key"]) starting from a given object.

        Args:
            obj: The object (e.g., dict, list, Pydantic model) to start resolution from.
            path_str: The dot-separated path string relative to 'obj'. 
                      Example: "my_list[0].name" or "my_dict[\"key-name\"].value"
            allow_partial: If True, returns None on any resolution error instead of raising.

        Returns:
            The resolved value.
        
        Raises:
            KeyError, AttributeError, IndexError, TypeError if resolution fails and allow_partial is False.
        """
        # self.logger.debug(f"_resolve_path_within_object: obj_type={type(obj)}, path_str='{path_str}'")
        current = obj
        
        if path_str is None: 
            self.logger.debug("Path string is None in _resolve_path_within_object.")
            if allow_partial: return None
            # If None path string is not allowed even with allow_partial, raise error:
            raise ValueError("Path string cannot be None for _resolve_path_within_object")

        # Robust segment parsing logic - now unconditional
        segments: List[str] = []
        current_segment = ""
        in_single_quotes = False
        in_double_quotes = False
        bracket_level = 0

        for char_idx, char in enumerate(path_str):
            is_escaped = char_idx > 0 and path_str[char_idx-1] == '\\'

            if char == "'" and not is_escaped and not in_double_quotes:
                in_single_quotes = not in_single_quotes
            elif char == '"' and not is_escaped and not in_single_quotes:
                in_double_quotes = not in_double_quotes
            
            if not in_single_quotes and not in_double_quotes:
                if char == '[':
                    bracket_level += 1
                elif char == ']':
                    if bracket_level > 0: # Ensure we don't decrement below zero from malformed input
                        bracket_level -= 1
            
                if char == '.' and bracket_level == 0:
                    segments.append(current_segment)
                    current_segment = ""
                    continue 
            current_segment += char
        segments.append(current_segment)
        
        parsed_accessors: List[str] = []
        for seg_idx, seg in enumerate(segments):
            # Handle cases like "obj." (empty last segment) or ".obj" (empty first segment)
            # An empty segment is only valid if it's the *only* segment from a path_str like "."
            if not seg:
                if len(segments) == 1 and path_str == ".": # Path is just "."
                    parsed_accessors.append(".") # Special case for "."
                    continue
                elif allow_partial and seg_idx < len(segments) -1 : # Empty segment in middle "a..b"
                     self.logger.warning(f"Empty segment encountered in path '{path_str}' at segment index {seg_idx}. Treating as None if allow_partial.")
                     return None # Or raise error if not allow_partial
                elif allow_partial and seg_idx == len(segments) -1 and path_str.endswith('.'): # Trailing dot "a."
                    self.logger.warning(f"Trailing dot in path '{path_str}'. Last segment empty.")
                    # Behavior for "a." might mean "a" itself, or error.
                    # If "a" was already added, this empty segment might be ignored or cause issues.
                    # For now, skip empty trailing segments.
                    continue
                elif not allow_partial :
                    raise ValueError(f"Empty segment in path '{path_str}' at index {seg_idx} not allowed.")
                else: # Skip if allow_partial and it's an empty segment not covered above
                    continue


            if '[' not in seg:
                parsed_accessors.append(seg)
            else:
                base_part = seg[:seg.find('[')]
                if base_part: # If there's a part before the first bracket (e.g., "my_list" in "my_list[0]")
                    parsed_accessors.append(base_part)
                
                bracket_substring = seg[seg.find('['):]
                # Correctly define the regex as a raw string.
                # This regex matches bracketed accessors like [0], ['key'], or ["key"].
                bracket_accessor_regex = re.compile(r"(\[(?:(?:\'[^\']*\')|(?:\"[^\"]*\")|(?:\d+))\])")
                found_bracket_accessors = False
                for match in bracket_accessor_regex.finditer(bracket_substring):
                    parsed_accessors.append(match.group(1)) 
                    found_bracket_accessors = True
                
                if not found_bracket_accessors and '[' in bracket_substring : # Check if there were brackets but regex didn't match
                    self.logger.error(f"Malformed bracket expression in segment: '{seg}' -> '{bracket_substring}'")
                    if allow_partial: return None
                    raise ValueError(f"Malformed bracket expression in path segment: {seg}")

        accessors = [acc for acc in parsed_accessors if acc is not None] # Filter out None, empty string "" can be a valid key.
                                                                        # Let's be more specific: filter if acc is truly empty only if not sole accessor from "."
        
        final_accessors = []
        for acc in accessors:
            if acc == "" and not (len(accessors) == 1 and path_str == "."): # Empty string accessor not allowed unless it's from path="."
                continue
            final_accessors.append(acc)
        accessors = final_accessors


        if not accessors:
            if path_str == "": return obj # Path was empty string, return current object
            # Path "." should result in accessors = ["."], _get_value_from_container should handle.
            # If path_str was non-empty but resulted in no accessors, it's an issue.
            self.logger.warning(f"Path string '{path_str}' resulted in no usable accessors after parsing. Original segments: {segments}, Parsed: {parsed_accessors}, Final: {accessors}")
            if allow_partial: return None
            # Avoid raising error if path_str itself implies returning the object (e.g. empty or ".")
            # which should be handled by _get_value_from_container or prior logic.
            # If execution reaches here with non-empty path_str and empty accessors, it's likely a parsing flaw.
            raise ValueError(f"Path '{path_str}' could not be parsed into effective accessors.")

        try:
            # self.logger.debug(f"Accessing with: {accessors} on object of type {type(current)}")
            return self._get_value_from_container(current, accessors)
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            # self.logger.debug(f"Resolution failed for '{path_str}' on {type(obj)}. Error: {e}")
            if allow_partial:
                return None
            else:
                # Augment error message with more context if possible
                # self.logger.error(f"Failed to resolve path '{path_str}' within object of type {type(obj)}. Last accessor attempted: {accessors[-1] if accessors else 'N/A'}. Error: {e}", exc_info=True)
                raise e # Re-raise the original error with its traceback

# Example Usage (Illustrative - would be used by Orchestrator)
if __name__ == '__main__':
    # This is for quick testing/dev, not part of the actual service use
    from chungoid.schemas.project_status_schema import SharedContext, ArtifactDetails, GlobalConfig

    # Setup mock logger
    mock_logger = logging.getLogger("ContextResolutionService_Test")
    mock_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    mock_logger.addHandler(handler)

    # Setup mock SharedContext
    sc = SharedContext(
        project_id="main_test_proj",
        run_id="main_test_run",
        project_root_path=Path("/dev/null"),
        outputs={
            "s1": {"out_key": "s1_value", "out_list": ["a", "b", {"c": "d"}], "dict-with-hyphen": "hyphenated_value"},
            "s2": {"data": {"nested_val": 123}},
        },
        artifacts={"art1": ArtifactDetails(path=Path("art1_path.txt"), metadata={"source": "gen"})},
        global_config=GlobalConfig(core_config={"timeout": 60}, tool_config={"my_tool": {"param_a": True}}),
        initial_inputs = {"initial_param": "hello_world"}
    )
    sc.previous_stage_outputs = {"prev_key": "previous_value"}

    # Instantiate the service
    resolver = ContextResolutionService(shared_context=sc, logger=mock_logger)

    # Test cases for resolve_inputs_for_stage
    inputs1 = {
        "val1": "{context.outputs.s1.out_key}",
        "val2": "{context.artifacts.art1.path}",
        "val3": "{context.global_config.core_config.timeout}",
        "val4": "{context.initial_inputs.initial_param}",
        "val5": "{context.outputs.s1.out_list[1]}", # Test list access
        "val6": "{context.outputs.s1.data.nested_val}", # Test nested dict access
        "literal_val": 100,
        "str_literal": "iamastring",
        "non_existent": "{context.outputs.s1.out_list[99]}",
        "bad_index": "{context.outputs.s1.out_list[5]}",
        "prev_out": "{context.previous_stage_outputs.prev_key}",
        "literal_str": "hello",
        "indexed_output": "{context.outputs.s1.out_list[2].c}",
        "quoted_key_output": "{context.outputs.s1[\"dict-with-hyphen\"]}"
    }
    mock_logger.info("--- Testing resolve_inputs_for_stage ---")
    resolved1 = resolver.resolve_inputs_for_stage(inputs1)
    mock_logger.info(f"Resolved inputs 1: {resolved1}")

    # Test cases for resolve_single_path
    paths_to_test = [
        "{context.outputs.s1.out_key}",
        "{context.artifacts.art1.metadata.source}",
        "{context.outputs.s1.out_list[0]}",
        "{context.outputs.s1.out_list[2]}",
        "{context.outputs.s1.data.nested_val}",
        "{context.outputs.s1.out_list[99]}", # out of bounds
        "{context.outputs.s1.out_list[bad_index_type]}", # invalid access on dict
        "{context.outputs.s1[\"dict-with-hyphen\"]}", # testing quoted key
        "{context.outputs.s1.out_list[0].non_attr}",       # None (attr on str)
        "{context.artifacts.art2}",                        # None
        "not_a_context_path",                              # None (malformed)
        "{context.outputs.s1.out_list[bad_index_type]}",   # None (malformed part)
        "{context.outputs.s1[\"dict-with-hyphen\"]}"         # hyphenated_value (testing quoted key)
    ]
    mock_logger.info("--- Testing resolve_single_path ---")
    for p in paths_to_test:
        resolved_single = resolver.resolve_single_path(p)
        mock_logger.info(f"Resolved single path '{p}': {resolved_single}") 