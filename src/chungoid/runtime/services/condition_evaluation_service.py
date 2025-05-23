from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional, Callable

from chungoid.runtime.services.context_resolution_service import ContextResolutionService
from chungoid.schemas.orchestration import SharedContext # Assuming this is where SharedContext is. Adjust if not.

class ConditionEvaluationError(Exception):
    """Custom exception for errors during condition evaluation."""
    pass

class ConditionEvaluationService:
    """
    Service responsible for parsing and evaluating condition strings
    used in orchestrator logic (e.g., for clarification checkpoints,
    conditional branching in future).
    """

    COMPARATOR_MAP: Dict[str, Callable[[Any, Any], bool]] = {
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "IN": lambda a, b: a in b,
        "NOT_IN": lambda a, b: a not in b,
        "CONTAINS": lambda a, b: b in a, # For checking if a string contains a substring, or list contains an element
        "NOT_CONTAINS": lambda a, b: b not in a,
        # TODO: Add more comparators as needed (e.g., regex match, STARTS_WITH, ENDS_WITH)
    }
    # Order matters for multi-character operators to avoid partial matches (e.g., > before >=)
    # Regex to capture LHS, operator, and RHS. Allows for quoted strings on RHS.
    # It captures: (LHS_path) (OPERATOR) (RHS_literal_or_path)
    # RHS can be quoted (single or double) or unquoted.
    CONDITION_REGEX = re.compile(r"^\s*([@{}.\\w\\-]+)\s*(" + "|".join(re.escape(op) for op in COMPARATOR_MAP.keys()) + r")\s*(?:(['\"])(.*?)\3|(.*?))\s*$")

    def __init__(self, context_resolver: ContextResolutionService, logger: Optional[logging.Logger] = None):
        self.context_resolver = context_resolver
        self.logger = logger or logging.getLogger(__name__)

    def _infer_literal_type(self, value_str: str) -> Any:
        """
        Infers the type of a literal string value.
        """
        value_str = value_str.strip()
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        if value_str.lower() == 'none' or value_str.lower() == 'null':
            return None
        # No need to check for quotes here if regex handles it, but keep for direct calls if any.
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        try:
            return int(value_str)
        except ValueError:
            pass
        try:
            return float(value_str)
        except ValueError:
            pass
        return value_str # Default to string

    def parse_and_evaluate_condition(
        self,
        condition_str: Optional[str],
        shared_context: SharedContext,
        current_stage_outputs: Optional[Dict[str, Any]] = None,
        # current_stage_name: Optional[str] = None # If needed for more specific context path resolution
    ) -> bool:
        """
        Parses a condition string and evaluates it against the provided context.

        Args:
            condition_str: The condition string to parse (e.g., "{context.some.value} == true").
                           If None or empty, evaluates to True (no condition).
            shared_context: The current shared context of the flow.
            current_stage_outputs: Outputs of the current stage, if applicable.

        Returns:
            True if the condition evaluates to true, False otherwise.

        Raises:
            ConditionEvaluationError: If the condition string is malformed or evaluation fails.
        """
        if not condition_str:
            self.logger.debug("No condition string provided, evaluating to True.")
            return True

        self.logger.debug(f"Parsing condition: '{condition_str}'")
        
        match = self.CONDITION_REGEX.match(condition_str)
        if not match:
            # Fallback for simple boolean path resolution (e.g., "{context.flags.should_run}")
            try:
                resolved_value = self.context_resolver.resolve_path_value_from_context(
                    path_str=condition_str.strip(),
                    shared_context=shared_context,
                    current_stage_outputs=current_stage_outputs,
                    # current_stage_name=current_stage_name # If needed
                    allow_partial=False
                )
                if isinstance(resolved_value, bool):
                    self.logger.debug(f"Condition '{condition_str}' resolved as boolean path to: {resolved_value}")
                    return resolved_value
                else:
                    self.logger.warning(f"Condition '{condition_str}' resolved to non-boolean value '{resolved_value}' ({type(resolved_value)}). Treating as malformed.")
                    raise ConditionEvaluationError(f"Condition '{condition_str}' resolved to a non-boolean value using direct path resolution.")
            except Exception as e: # Catch resolution errors specifically if needed
                self.logger.error(f"Malformed condition string or path resolution error for '{condition_str}': {e}")
                raise ConditionEvaluationError(f"Malformed condition string or path resolution error: '{condition_str}'. Error: {e}") from e

        lhs_path = match.group(1)
        operator = match.group(2)
        # Group 4 is for quoted string content, group 5 for unquoted literal
        rhs_literal_str = match.group(4) if match.group(4) is not None else match.group(5)

        if rhs_literal_str is None:
             raise ConditionEvaluationError(f"Could not parse RHS in condition: '{condition_str}'")

        try:
            lhs_value = self.context_resolver.resolve_path_value_from_context(
                path_str=lhs_path,
                shared_context=shared_context,
                current_stage_outputs=current_stage_outputs,
                # current_stage_name=current_stage_name # If needed
                allow_partial=False # Conditions usually require full resolution
            )
        except Exception as e:
            self.logger.error(f"Could not resolve LHS path '{lhs_path}' in condition '{condition_str}': {e}")
            raise ConditionEvaluationError(f"Could not resolve LHS path '{lhs_path}' in condition '{condition_str}': {e}") from e

        # RHS can be a literal or another path. For now, assume literal and infer type.
        # If RHS could also be a path, this logic would need to try path resolution first.
        if match.group(3): # If group 3 (quote char) is present, it was a quoted string.
            rhs_value = rhs_literal_str # It's already a string from the regex
        else: # Unquoted literal, infer type
            rhs_value = self._infer_literal_type(rhs_literal_str)
        
        comparator_func = self.COMPARATOR_MAP.get(operator)
        if not comparator_func:
            # This should not happen if regex is synced with COMPARATOR_MAP keys
            raise ConditionEvaluationError(f"Unsupported operator '{operator}' in condition '{condition_str}'. Should have been caught by regex.")

        try:
            result = comparator_func(lhs_value, rhs_value)
            self.logger.debug(f"Condition '{condition_str}' (LHS: '{lhs_path}' -> {lhs_value} ({type(lhs_value)}), Op: {operator}, RHS: '{rhs_literal_str}' -> {rhs_value} ({type(rhs_value)})) evaluated to {result}")
            return result
        except TypeError as te:
            self.logger.warning(f"Type error during condition evaluation for '{condition_str}' (LHS: {lhs_value} ({type(lhs_value)}), Op: {operator}, RHS: {rhs_value} ({type(rhs_value)})): {te}. Returning False.")
            # Depending on strictness, could raise error or return False.
            # For now, returning False as conditions failing due to type errors are often intended to be false.
            return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition_str}': {e}", exc_info=True)
            raise ConditionEvaluationError(f"Error evaluating condition '{condition_str}': {e}") from e 