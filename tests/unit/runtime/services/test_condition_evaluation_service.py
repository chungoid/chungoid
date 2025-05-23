from __future__ import annotations

import pytest
import logging
from unittest.mock import MagicMock, patch
from typing import Optional, Dict, Any

from chungoid.runtime.services.condition_evaluation_service import ConditionEvaluationService, ConditionEvaluationError
from chungoid.runtime.services.context_resolution_service import ContextResolutionService
from chungoid.schemas.orchestration import SharedContext
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec

# Minimal MasterExecutionPlan for SharedContext initialization
SAMPLE_PLAN = MasterExecutionPlan(
    id="test_plan",
    name="Test Plan",
    description="A test plan",
    start_stage="stage1",
    stages={"stage1": MasterStageSpec(id="stage1", name="Test Stage 1", agent_id="test_agent")}
)

@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=logging.Logger)
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger

@pytest.fixture
def mock_context_resolver(mock_logger):
    resolver = MagicMock(spec=ContextResolutionService)
    resolver.logger = mock_logger
    # Mock the methods that ConditionEvaluationService might call
    resolver.resolve_path_value_from_context = MagicMock(
        side_effect=lambda path_str, shared_ctx, current_stage_outputs=None, **kwargs: \
            _mock_resolver_side_effect(path_str, shared_ctx, current_stage_outputs=current_stage_outputs, **kwargs)
    )
    resolver.resolve_single_path = MagicMock(
        side_effect=lambda path_str, shared_ctx, **kwargs: \
            _mock_resolver_side_effect(path_str, shared_ctx, is_single_path_call=True, **kwargs)
    )    
    return resolver

@pytest.fixture
def shared_context_fixture():
    ctx = SharedContext(
        flow_id="test_flow",
        run_id="test_run",
        current_master_plan=SAMPLE_PLAN,
        initial_inputs = {"key1": "value1", "number": 10, "flag_true": True, "flag_false": False, "nested": {"key2": "value2"}, "none_val": None, "str_list": "item1,item2"},
        previous_stage_outputs = {"prev_stage": {"output_val": "abc"}},
        artifact_references = {"artifact1": "/path/to/artifact1.txt"}
    )
    return ctx

@pytest.fixture
def condition_service(mock_context_resolver, mock_logger):
    return ConditionEvaluationService(context_resolver=mock_context_resolver, logger=mock_logger)

# --- Test Cases ---

def test_evaluate_empty_or_none_condition(condition_service, shared_context_fixture):
    assert condition_service.parse_and_evaluate_condition(None, shared_context_fixture) is True
    assert condition_service.parse_and_evaluate_condition("", shared_context_fixture) is True
    # A string with only spaces will try to be resolved as a path, fail, and raise ConditionEvaluationError
    with pytest.raises(ConditionEvaluationError):
        condition_service.parse_and_evaluate_condition("   ", shared_context_fixture)

def test_evaluate_simple_boolean_path_true(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = True
    assert condition_service.parse_and_evaluate_condition("{context.flag_true}", shared_context_fixture) is True
    mock_context_resolver.resolve_path_value_from_context.assert_called_once_with(
        path_str="{context.flag_true}",
        shared_context=shared_context_fixture,
        current_stage_outputs=None,
        allow_partial=False
    )

def test_evaluate_simple_boolean_path_false(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = False
    assert condition_service.parse_and_evaluate_condition("{context.flag_false}", shared_context_fixture) is False

def test_evaluate_simple_boolean_path_non_boolean_value_raises_error(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = "not_a_boolean"
    with pytest.raises(ConditionEvaluationError, match="resolved to a non-boolean value using direct path resolution"):
        condition_service.parse_and_evaluate_condition("{context.key1}", shared_context_fixture)

def test_malformed_condition_direct_path_resolution_failure_raises_error(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.side_effect = ValueError("Resolution failed")
    with pytest.raises(ConditionEvaluationError, match="Malformed condition string or path resolution error"):
        condition_service.parse_and_evaluate_condition("just_a_string_no_operator", shared_context_fixture)


@pytest.mark.parametrize("condition_str, lhs_resolved_value, expected_result", [
    # Equals (==)
    ("{context.key1} == value1", "value1", True),
    ("{context.key1} == value2", "value1", False),
    ("{context.number} == 10", 10, True),
    ("{context.number} == 10.0", 10, True),
    ("{context.flag_true} == true", True, True),
    ("{context.flag_false} == false", False, True),
    ("{context.flag_true} == false", True, False),
    ("{context.nested.key2} == value2", "value2", True),
    ("{context.none_val} == none", None, True),
    # Not Equals (!=)
    ("{context.key1} != value2", "value1", True),
    ("{context.number} != 10", 10, False),
    ("{context.flag_true} != false", True, True),
    ("{context.none_val} != none", "some_value", True),
    # Greater Than (>)
    ("{context.number} > 5", 10, True),
    ("{context.number} > 10", 10, False),
    # Less Than (<)
    ("{context.number} < 15", 10, True),
    ("{context.number} < 10", 10, False),
    # Greater Than or Equals (>=)
    ("{context.number} >= 10", 10, True),
    ("{context.number} >= 5", 10, True),
    ("{context.number} >= 11", 10, False),
    # Less Than or Equals (<=)
    ("{context.number} <= 10", 10, True),
    ("{context.number} <= 15", 10, True),
    ("{context.number} <= 9", 10, False),
    # CONTAINS (string specific for now)
    ("{context.key1} CONTAINS lue", "value1", True),
    ("{context.key1} CONTAINS xyz", "value1", False),
    # NOT_CONTAINS (string specific for now)
    ("{context.key1} NOT_CONTAINS xyz", "value1", True),
    ("{context.key1} NOT_CONTAINS lue", "value1", False),
    # IN (specific string in string for now, not ideal but tests basic path)
    ("{context.key1} IN value1_or_value2", "value1", True), # RHS treated as literal string "value1_or_value2"
    ("{context.number} IN 10_or_20", 10, True), # RHS treated as literal string "10_or_20"
    # NOT_IN (similar to IN)
    ("{context.key1} NOT_IN not_value1", "value1", True),
    ("{context.number} NOT_IN not_10", 10, True),
])
def test_various_operator_conditions(condition_service, shared_context_fixture, mock_context_resolver,
                                     condition_str, lhs_resolved_value, expected_result):
    lhs_path = condition_str.split(" ")[0]
    mock_context_resolver.resolve_path_value_from_context.reset_mock() # Reset mock for each parametrized call
    mock_context_resolver.resolve_path_value_from_context.return_value = lhs_resolved_value
    
    result = condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture)
    assert result == expected_result
    mock_context_resolver.resolve_path_value_from_context.assert_called_with(
        path_str=lhs_path,
        shared_context=shared_context_fixture,
        current_stage_outputs=None,
        allow_partial=False
    )

def test_type_error_during_evaluation_returns_false(condition_service, shared_context_fixture, mock_context_resolver, mock_logger):
    mock_context_resolver.resolve_path_value_from_context.return_value = 10 # LHS is int
    condition_str = "{context.number} == NonCoercibleString" 
    assert condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture) is False
    mock_logger.warning.assert_called()
    assert "Type error during condition evaluation" in mock_logger.warning.call_args[0][0]

def test_lhs_path_resolution_failure_raises_error(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.side_effect = ValueError("Failed to resolve path")
    condition_str = "{context.nonexistent} == true"
    with pytest.raises(ConditionEvaluationError, match="Could not resolve LHS path"):
        condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture)

def test_unsupported_operator_if_regex_somehow_bypassed(condition_service, shared_context_fixture, mock_context_resolver):
    # This tests the direct malformed condition path if regex somehow fails for an unknown operator pattern.
    condition_str = "{context.key1} UNKNOWN_OPERATOR value1"
    with pytest.raises(ConditionEvaluationError, match="Malformed condition string or path resolution error"):
        condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture)

def test_condition_with_current_stage_outputs(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = "output_match"
    current_outputs = {"current_val": "output_match"}
    condition_str = "{@current_stage.current_val} == output_match"
    assert condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture, current_stage_outputs=current_outputs) is True
    mock_context_resolver.resolve_path_value_from_context.assert_called_with(
        path_str="{@current_stage.current_val}",
        shared_context=shared_context_fixture,
        current_stage_outputs=current_outputs,
        allow_partial=False
    )

def test_infer_literal_type(condition_service):
    assert condition_service._infer_literal_type("true") is True
    assert condition_service._infer_literal_type("FALSE") is False
    assert condition_service._infer_literal_type("None") is None
    assert condition_service._infer_literal_type("null") is None
    assert condition_service._infer_literal_type("123") == 123
    assert condition_service._infer_literal_type("123.45") == 123.45
    assert condition_service._infer_literal_type("\"hello world\"") == "hello world" # String with double quotes
    assert condition_service._infer_literal_type("'quoted string'") == "quoted string" # String with single quotes
    assert condition_service._infer_literal_type("plain string") == "plain string"
    assert condition_service._infer_literal_type("  true  ") is True 
    assert condition_service._infer_literal_type("  \"  quoted with spaces  \"  ") == "  quoted with spaces  "
    assert condition_service._infer_literal_type("") == ""
    assert condition_service._infer_literal_type("  ") == ""

def test_in_operator_type_error_logged(condition_service, shared_context_fixture, mock_context_resolver, mock_logger):
    mock_context_resolver.resolve_path_value_from_context.return_value = 10 # LHS is int
    condition_str = "{context.number} IN [10, 20, 30]" # RHS is string "[10, 20, 30]"
    # `10 in "[10, 20, 30]"` would cause a TypeError for the `in` operator if not handled.
    # The service currently catches TypeError and logs a warning, then returns False.
    assert condition_service.parse_and_evaluate_condition(condition_str, shared_context_fixture) is False
    mock_logger.warning.assert_called()
    assert "Type error during condition evaluation" in mock_logger.warning.call_args[0][0]

def test_edge_case_rhs_none_explicitly(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = None
    assert condition_service.parse_and_evaluate_condition("{context.none_val} == none", shared_context_fixture) is True

    mock_context_resolver.resolve_path_value_from_context.return_value = "not_none"
    assert condition_service.parse_and_evaluate_condition("{context.not_none} == none", shared_context_fixture) is False

    mock_context_resolver.resolve_path_value_from_context.return_value = None
    assert condition_service.parse_and_evaluate_condition("{context.none_val} != none", shared_context_fixture) is False

# Further tests could explore RHS as path, more complex list/dict structures for IN/CONTAINS once service supports them.

# More tests needed for:
# - Edge cases in _infer_literal_type (e.g., empty strings, strings that look like numbers but aren't)
# - More complex path expressions for LHS if supported by ContextResolutionService
# - Conditions where RHS is also a path (if that becomes a feature)
# - Specific behaviors for IN, NOT_IN, CONTAINS, NOT_CONTAINS with various RHS types (e.g. string representation of list vs actual list)
# - Case sensitivity for operators or literals (currently, operators are case-sensitive, string literals are exact)

# Tests for list/array-like string literals for IN and NOT_IN (if _infer_literal_type is not expected to handle them)
# These might require ConditionEvaluationService.parse_and_evaluate_condition to have special logic
# for these operators to parse the RHS string (e.g. "['a','b']" or "'a','b','c'") into a list.
# The current COMPARATOR_MAP lambdas for IN/NOT_IN expect RHS to be an actual list/iterable.

def test_in_operator_with_list_like_string_rhs(condition_service, shared_context_fixture, mock_context_resolver):
    # This test highlights a limitation: _infer_literal_type doesn't parse "['a','b']" into a list.
    # The `IN` lambda `a in b` would try `resolved_lhs_value in "['value1', 'value2']"`.
    # This might work if LHS is a single char and RHS is a string, but not for general element-in-list.
    
    # Scenario 1: LHS is a string, RHS is a string that looks like a list.
    # `_infer_literal_type` will return the RHS as a string "['item1', 'item2']".
    # The `IN` operator `lambda a, b: a in b` would become ` "item1" in "['item1', 'item2']" `
    # This specific string check would be True. This might be acceptable for some use cases.
    mock_context_resolver.resolve_path_value_from_context.return_value = "item1"
    condition_str_list_like = "{context.data} IN ['item1', 'item2']" # RHS is a string literal here
    assert condition_service.parse_and_evaluate_condition(condition_str_list_like, shared_context_fixture) is True

    # Scenario 2: LHS is a number, RHS is a string that looks like a list of numbers.
    # `10 in "[10, 20]" ` will be False because 10 is not a substring of "[10, 20]" in that way.
    mock_context_resolver.resolve_path_value_from_context.return_value = 10
    condition_str_num_list_like = "{context.number} IN [10, 20, 30]"
    # This will likely be False because `_infer_literal_type` returns "[10, 20, 30]" as a string.
    # `10 in "[10, 20, 30]"` is False.
    # To make this True, parse_and_evaluate_condition would need to specifically parse the RHS for IN/NOT_IN
    assert condition_service.parse_and_evaluate_condition(condition_str_num_list_like, shared_context_fixture) is False
    mock_logger.warning.assert_called_once() # Expect a TypeError because int `in` str is not directly useful for element check
    assert "Type error during condition evaluation" in mock_logger.warning.call_args[0][0]


    # If the service were to parse RHS for IN/NOT_IN:
    # with patch.object(condition_service, '_parse_rhs_for_in_operator', return_value=['item1', 'item2']) as mock_parse_rhs:
    #     mock_context_resolver.resolve_path_value_from_context.return_value = "item1"
    #     condition_str_list_like = "{context.data} IN ['item1', 'item2']"
    #     assert condition_service.parse_and_evaluate_condition(condition_str_list_like, shared_context_fixture) is True
    #     mock_parse_rhs.assert_called_once_with("['item1', 'item2']")

def test_edge_case_rhs_none_explicitly(condition_service, shared_context_fixture, mock_context_resolver):
    mock_context_resolver.resolve_path_value_from_context.return_value = None
    assert condition_service.parse_and_evaluate_condition("{context.maybe_none} == none", shared_context_fixture) is True

    mock_context_resolver.resolve_path_value_from_context.return_value = "not_none"
    assert condition_service.parse_and_evaluate_condition("{context.not_none} == none", shared_context_fixture) is False

    mock_context_resolver.resolve_path_value_from_context.return_value = None
    assert condition_service.parse_and_evaluate_condition("{context.maybe_none} != none", shared_context_fixture) is False

def _mock_resolver_side_effect(path_str: str, shared_ctx: SharedContext, current_stage_outputs: Optional[Dict[str,Any]] = None, is_single_path_call: bool = False, allow_partial: bool = False, **kwargs):
    # More robust mock for path resolution
    normalized_path = path_str
    
    # Handle {@current_stage...} paths
    if normalized_path.startswith("{@current_stage.") and normalized_path.endswith("}"):
        key_path = normalized_path[len("{@current_stage."):-1]
        # Resolve key_path within current_stage_outputs
        if current_stage_outputs:
            current_val = current_stage_outputs
            for part in key_path.split("."):
                if isinstance(current_val, dict) and part in current_val:
                    current_val = current_val[part]
                elif hasattr(current_val, part): # For Pydantic models
                    current_val = getattr(current_val, part)
                else:
                    raise KeyError(f"Mock: Path part '{part}' not found in current_stage_outputs for '{path_str}'.")
            return current_val
        else:
            raise KeyError(f"Mock: current_stage_outputs is None, cannot resolve '{path_str}'.")

    # Handle {context...} paths
    elif normalized_path.startswith("{context.") and normalized_path.endswith("}"):
        key_path = normalized_path[len("{context."):-1]
    # Handle context... paths (for resolve_single_path from ConditionEvaluationService)
    elif normalized_path.startswith("context.") and is_single_path_call: # path comes as "context.some.path"
        key_path = normalized_path[len("context."):]
    # Handle direct paths if wrapped in {}
    elif normalized_path.startswith("{") and normalized_path.endswith("}"):
        key_path = normalized_path[1:-1]
    else:
        # If it's a plain string without operators, it's treated as a direct path by ConditionEvaluationService
        # The service will attempt to resolve it. If it's not a special format, this mock will try to look it up as is.
        key_path = normalized_path

    # Resolve key_path within shared_context.initial_inputs (common for these tests)
    # or other parts of shared_context if necessary for more complex tests.
    # This simplified version primarily checks initial_inputs.
    
    parts = key_path.split('.')
    current_val = shared_ctx.initial_inputs # Start with initial_inputs as per most test data
    
    if key_path == "key1": return shared_ctx.initial_inputs.get("key1")
    if key_path == "number": return shared_ctx.initial_inputs.get("number")
    if key_path == "flag_true": return shared_ctx.initial_inputs.get("flag_true")
    if key_path == "flag_false": return shared_ctx.initial_inputs.get("flag_false")
    if key_path == "nested.key2":
        return shared_ctx.initial_inputs.get("nested", {}).get("key2")
    if key_path == "none_val": return shared_ctx.initial_inputs.get("none_val") # Handles explicit None
    if key_path == "str_list": return shared_ctx.initial_inputs.get("str_list")
    
    # Try to traverse shared_context attributes for more general paths if not found above
    current_search_obj = shared_ctx
    resolved = False
    for part_idx, part in enumerate(parts):
        if hasattr(current_search_obj, part):
            current_search_obj = getattr(current_search_obj, part)
            if part_idx == len(parts) - 1: # Last part
                resolved = True
                break
        elif isinstance(current_search_obj, dict) and part in current_search_obj:
            current_search_obj = current_search_obj[part]
            if part_idx == len(parts) - 1: # Last part
                resolved = True
                break
        else: # Path segment not found
            break 
            
    if resolved:
        return current_search_obj

    # Fallback for paths not explicitly handled (e.g., nonexistent paths for error testing)
    if "nonexistent" in key_path or not resolved : # Path not found
        if allow_partial: # If partial resolution is allowed, and we resolved something
            if current_search_obj != shared_ctx : # if we moved from the root shared_ctx
                # This logic is tricky for a simple mock.
                # For now, if allow_partial and not fully resolved, act like full failure for simplicity in tests.
                 raise KeyError(f"Mock: Path '{path_str}' (normalized to '{key_path}') not fully found in mock resolver, even with allow_partial=True.")
        raise KeyError(f"Mock: Path '{path_str}' (normalized to '{key_path}') not found in mock resolver.")

    return current_search_obj # Should be the resolved value if path was simple and directly in initial_inputs initially 