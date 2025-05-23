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
            if not shared_context.project_status or not hasattr(shared_context.project_status, 'artifacts'):
                raise ValueError("project_status.artifacts not found in SharedContext")
            
            artifact = shared_context.project_status.artifacts.get(item_id)
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
                    if shared_context.project_root_path is None:
                        raise ValueError(
                            f"project_root_path must be set in SharedContext to resolve relative artifact content path: {file_path}"
                        )
                    file_path = shared_context.project_root_path / file_path
                
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
        Recursively (iteratively) gets a value from a nested structure (dict, list, object)
        using a list of accessors (attribute names, dict keys, or list indices as strings like "[0]").
        """
        current = container
        for accessor in accessors:
            # Clean accessor if it's a quoted key like ["key-name"] or ['key-name']
            # This step is crucial because the regex might pass these through.
            # Regex captures: .key, [\'key\'], [\"key\"], [0]
            # We need to ensure that when current is a dict, we use the unquoted key.
            # For getattr, the original accessor (if it's a valid attribute name) is fine.

            is_list_index = False
            actual_accessor = accessor # Default to using it as is (for getattr)

            if isinstance(accessor, str) and accessor.startswith('['):
                inner_accessor = accessor[1:-1]
                if inner_accessor.isdigit(): # Integer index like [0]
                    is_list_index = True
                    actual_accessor = int(inner_accessor)
                elif (inner_accessor.startswith("'") and inner_accessor.endswith("'")) or \
                     (inner_accessor.startswith('"') and inner_accessor.endswith('"')): # Quoted key like ["my-key"]
                    actual_accessor = inner_accessor[1:-1] # Unquote for dict key access
                else:
                    # This case should ideally not be hit if the regex is precise and
                    # subsequent logic correctly parses its groups.
                    # It implies an unquoted string within brackets not matching other patterns, e.g., [my_key_without_quotes]
                    # which is ambiguous. For safety, we might treat it as a potential dict key.
                    actual_accessor = inner_accessor

            if is_list_index:
                if not isinstance(current, list):
                    self.logger.error(f"Attempted to index a non-list type ({type(current)}) with index '{actual_accessor}'. Path part: {accessor}")
                    raise TypeError(f"Attempted to index a non-list type ({type(current)}) with index '{actual_accessor}'. Path part: {accessor}")
                try:
                    current = current[actual_accessor]
                except IndexError:
                    self.logger.error(f"List index {actual_accessor} out of range for list of length {len(current)}. Path part: {accessor}")
                    raise IndexError(f"List index {actual_accessor} out of range. Path part: {accessor}")
            elif isinstance(current, dict):
                # Use the potentially unquoted actual_accessor for dict key lookup
                if actual_accessor not in current:
                    self.logger.error(f"Key '{actual_accessor}' not found in dictionary {list(current.keys())}. Path part: {accessor}")
                    raise KeyError(f"Key '{actual_accessor}' not found in dictionary. Path part: {accessor}")
                current = current[actual_accessor]
            elif isinstance(current, BaseModel): # ADDED Pydantic BaseModel check
                if hasattr(current, accessor):
                    current = getattr(current, accessor)
                else:
                    try:
                        current_dump = current.model_dump()
                        if accessor in current_dump:
                            current = current_dump[accessor]
                        else:
                            self.logger.error(f"Attribute/field '{accessor}' not found on Pydantic model {type(current)} via direct access or in model_dump(). Path part: {accessor}")
                            raise AttributeError(f"Attribute/field '{accessor}' not found on Pydantic model {type(current)}. Path part: {accessor}")
                    except Exception as e_pydantic_fallback:
                        self.logger.error(f"Error during Pydantic model_dump() fallback for '{accessor}' on {type(current)}: {e_pydantic_fallback}. Path part: {accessor}")
                        raise AttributeError(f"Attribute/field '{accessor}' not found on Pydantic model {type(current)} or fallback failed. Path part: {accessor}")
            else: # General object attribute access
                # Use original accessor for getattr
                if not hasattr(current, accessor):
                    self.logger.error(f"Object of type {type(current)} has no attribute '{accessor}'. Path part: {accessor}")
                    raise AttributeError(f"Object of type {type(current)} has no attribute '{accessor}'. Path part: {accessor}")
                current = getattr(current, accessor)
        return current

    def resolve_single_path(
        self,
        path_expression: str,
        shared_context: Optional[SharedContext] = None,
        allow_partial: bool = False # ADDED allow_partial
    ) -> Any:
        """
        Resolves a single dot-separated path expression against the shared context.
        Example: "{context.outputs.stage_one.keyA}" or "@artifact.my_id.path"
        Handles direct attribute access, dictionary key access (including quoted keys), and list indexing.
        """
        current_shared_context = shared_context if shared_context else self.shared_context
        if not current_shared_context:
            self.logger.warning("SharedContext is not available for path resolution.")
            # If allow_partial is true, we might want to return a specific marker or the original path
            # For now, returning None aligns with previous behavior for missing context.
            return None 

        original_path_expression = path_expression # Keep for logging

        # Normalize: strip {}
        clean_path = path_expression
        if path_expression.startswith("{") and path_expression.endswith("}"):
            clean_path = path_expression[1:-1]
        
        if not clean_path:
            self.logger.warning(f"Path expression '{original_path_expression}' became empty after cleaning.")
            return None

        # self.logger.debug(f"Resolving single path: '{clean_path}' (original: '{original_path_expression}')")

        if clean_path.startswith("@"):
            try:
                return self._resolve_at_path(clean_path, current_shared_context)
            except Exception as e:
                self.logger.warning(f"Failed to resolve @path '{clean_path}': {e}")
                if allow_partial: return None # Or path_expression if we want to return unresolved paths
                raise # Re-raise if not allowing partial resolution to signal critical failure

        path_parts = clean_path.split('.', 1)
        base_name = path_parts[0]
        remaining_access_path_str = path_parts[1] if len(path_parts) > 1 else None
        
        current_value: Any
        
        # Determine the base object from SharedContext
        if base_name == "context":
            # If path is just "{context}", return the SharedContext object itself (or its data representation)
            if not remaining_access_path_str:
                # Decide what "{context}" itself should resolve to. 
                # Returning the .data attribute seems more useful for templating than the object itself.
                return current_shared_context.data if hasattr(current_shared_context, 'data') else current_shared_context

            # Standardize to start processing from current_shared_context.data if path implies structured data access
            # Path examples: context.outputs.stage.key, context.data.project_id, context.project_id (direct attribute)
            
            # Check for direct attributes on SharedContext first (e.g., project_id, run_id)
            # These should take precedence if they are not dictionary keys under .data
            first_part_of_remaining = remaining_access_path_str.split('.', 1)[0]
            
            is_direct_attribute_on_context = hasattr(current_shared_context, first_part_of_remaining)
            is_key_in_context_data = (
                hasattr(current_shared_context, 'data') and 
                isinstance(current_shared_context.data, dict) and 
                first_part_of_remaining in current_shared_context.data
            )

            if is_direct_attribute_on_context and not is_key_in_context_data:
                # If it's a direct attribute of SharedContext object and NOT also a key in its data dict,
                # then resolve starting from the SharedContext object itself.
                current_value = current_shared_context
                # remaining_access_path_str is already correctly set for _resolve_path_within_object
            elif first_part_of_remaining == "data" and hasattr(current_shared_context, 'data'):
                current_value = current_shared_context.data
                # Strip "data." from the remaining path
                parts = remaining_access_path_str.split('.', 1)
                if len(parts) > 1:
                    remaining_access_path_str = parts[1]
                else: # Path was just "context.data"
                    remaining_access_path_str = None # current_value (the data dict) will be returned
            elif hasattr(current_shared_context, 'data'): # Default to data for other context paths if not direct attr
                current_value = current_shared_context.data
                # remaining_access_path_str is already set correctly for _resolve_path_within_object
                # This case might now be less common if 'data.' prefix is handled above,
                # but handles cases like "{context.some_key_directly_in_data_dict}"
            else:
                self.logger.warning(f"SharedContext has no 'data' attribute, or path 'context.{first_part_of_remaining}' is ambiguous/unresolvable for path '{original_path_expression}'.")
                if allow_partial: return None
                raise AttributeError(f"SharedContext attribute or data key '{first_part_of_remaining}' issue for path '{original_path_expression}'")
            
            # remaining_access_path_str is already set correctly for _resolve_path_within_object

        elif base_name == "outputs": # For shorthand like {outputs.stage.key}
            if not hasattr(current_shared_context, 'data') or not isinstance(current_shared_context.data.get("outputs"), dict):
                self.logger.warning(f"'data[\"outputs\"]' not found or not a dict on SharedContext for path: {original_path_expression}")
                if allow_partial: return None
                raise AttributeError("'data[\"outputs\"]' not found or not a dict on SharedContext")
            current_value = current_shared_context.data["outputs"]
            # remaining_access_path_str is already set

        elif base_name == "global_config": # For {global_config.some_key}
            if not hasattr(current_shared_context, 'data') or not isinstance(current_shared_context.data.get("global_config"), dict):
                self.logger.warning(f"'data[\"global_config\"]' not found or not a dict on SharedContext for path: {original_path_expression}")
                if allow_partial: return None
                raise AttributeError("'data[\"global_config\"]' not found or not a dict on SharedContext")
            current_value = current_shared_context.data["global_config"]
            # remaining_access_path_str is already set

        # Add other top-level namespaces if necessary, e.g., "inputs", "settings"
        # For now, assume other base_names might be custom top-level keys in shared_context.data
        elif hasattr(current_shared_context, 'data') and isinstance(current_shared_context.data, dict) and base_name in current_shared_context.data:
            current_value = current_shared_context.data[base_name]
            # remaining_access_path_str is already set
        else:
            self.logger.warning(f"Base path element '{base_name}' not found in SharedContext or its data for path: {original_path_expression}")
            if allow_partial: return None
            raise KeyError(f"Base path element '{base_name}' not found in SharedContext for '{original_path_expression}'")

        if remaining_access_path_str is None:
            # Path was just "{context}" or "{outputs}" - return the base object determined above
            return current_value 
        
        try:
            # self.logger.debug(f"Calling _resolve_path_within_object with obj: {type(current_value)}, path_str: '{remaining_access_path_str}'")
            return self._resolve_path_within_object(current_value, remaining_access_path_str, allow_partial=allow_partial)
        except (KeyError, AttributeError, IndexError, TypeError) as e:
            self.logger.warning(f"Could not resolve path '{remaining_access_path_str}' within the identified base object for '{original_path_expression}'. Error: {e}")
            if allow_partial: return None # Or path_expression
            raise

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