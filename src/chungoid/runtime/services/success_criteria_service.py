"""
Service for evaluating success criteria for orchestrator stages.
"""
import logging
import re
from typing import List, Dict, Any, Tuple, Optional

from chungoid.schemas.master_flow import MasterStageSpec
from chungoid.schemas.orchestration import SharedContext
from chungoid.runtime.services.context_resolution_service import ContextResolutionService

# Sentinel object for cases where a path is not found during resolution
_SENTINEL = object()

logger = logging.getLogger(__name__)

class SuccessCriteriaService:
    """
    Evaluates success criteria defined in a MasterStageSpec against stage outputs
    and shared context.
    """
    def __init__(self, logger: logging.Logger, context_resolver: ContextResolutionService):
        self.logger = logger
        self.context_resolver = context_resolver

    async def _resolve_value_for_criterion(
        self,
        path_str: str,
        stage_outputs: Any,
        shared_context_for_stage: SharedContext
    ) -> Any:
        """
        Resolves a path string against stage_outputs or shared_context.
        Priority is given to stage_outputs if path doesn't explicitly target context.
        """
        self.logger.debug(f"Attempting to resolve path for criterion: '{path_str}'")

        if path_str.startswith("{context.") and path_str.endswith("}"):
            # Explicit context path
            path_expression_in_braces = path_str[len("{context."):-1]
            try:
                resolved_value = await self.context_resolver.resolve_single_path(
                    path_expression_in_braces, shared_context_for_stage
                )
                self.logger.debug(f"Resolved '{path_str}' from shared_context to: {resolved_value}")
                return resolved_value
            except Exception as e:
                self.logger.error(f"Unexpected error resolving context path '{path_str}': {e}", exc_info=True)
                return _SENTINEL
        
        current_obj = stage_outputs
        parts = path_str.split('.')
        
        if path_str.startswith("outputs.") and isinstance(stage_outputs, dict):
            actual_path_in_outputs = path_str[len("outputs."):]
            parts = actual_path_in_outputs.split('.')

        for part in parts:
            if isinstance(current_obj, dict):
                if part in current_obj:
                    current_obj = current_obj[part]
                else:
                    self.logger.debug(f"Path part '{part}' not found in dict during criterion path resolution for '{path_str}'.")
                    return _SENTINEL
            elif hasattr(current_obj, part): 
                current_obj = getattr(current_obj, part)
            else:
                self.logger.debug(f"Path part '{part}' not found as attribute or key during criterion path resolution for '{path_str}'.")
                return _SENTINEL
        
        self.logger.debug(f"Resolved '{path_str}' from stage_outputs (or direct access) to: {current_obj}")
        return current_obj

    async def check_criteria_for_stage(
        self,
        stage_name: str,
        stage_spec: MasterStageSpec,
        stage_outputs: Any, 
        shared_context_for_stage: SharedContext 
    ) -> Tuple[bool, List[str]]:
        """
        Checks all success_criteria for a given stage.
        """
        if not stage_spec.success_criteria:
            self.logger.info(f"No success criteria defined for stage '{stage_name}'. Defaulting to success.")
            return True, []

        all_passed = True
        failed_criteria: List[str] = []

        self.logger.info(
            f"Checking {len(stage_spec.success_criteria)} success criteria for stage '{stage_name}'."
        )
        for criterion_str in stage_spec.success_criteria:
            if not await self._evaluate_single_criterion(
                criterion_str=criterion_str,
                stage_name=stage_name,
                stage_outputs=stage_outputs,
                shared_context_for_stage=shared_context_for_stage
            ):
                self.logger.warning(f"Stage '{stage_name}' failed success criterion: {criterion_str}")
                failed_criteria.append(criterion_str)
                all_passed = False
        
        if not all_passed:
            self.logger.warning(
                f"Stage '{stage_name}' failed one or more success criteria. Failed: {failed_criteria}"
            )
        else:
            self.logger.info(f"All success criteria passed for stage '{stage_name}'.")
        return all_passed, failed_criteria

    async def _evaluate_single_criterion(
        self,
        criterion_str: str,
        stage_name: str,
        stage_outputs: Any, 
        shared_context_for_stage: SharedContext
    ) -> bool:
        self.logger.debug(
            f"Evaluating criterion: '{criterion_str}' for stage '{stage_name}'"
        )

        operators = {
            "EXISTS": 1, "IS_NOT_EMPTY": 1, "IS_EMPTY": 1, "IS_TRUE": 1,
            "IS_FALSE": 1, "IS_NONE": 1, "IS_NOT_NONE": 1, "CONTAINS": 2,
            ">=": 2, "<=": 2, "!=": 2, "==": 2, ">": 2, "<": 2,
        }

        lhs_str = ""
        op = ""
        rhs_str: Optional[str] = None
        
        parsed_successfully = False
        for op_key in operators:
            temp_parts = criterion_str.split(op_key, 1)
            if len(temp_parts) > 1: 
                lhs_str = temp_parts[0].strip()
                op = op_key
                if operators[op_key] == 2:
                    rhs_str = temp_parts[1].strip()
                parsed_successfully = True
                break
        
        if not parsed_successfully and " " in criterion_str: 
            parts = criterion_str.rsplit(" ", 1) 
            potential_op = parts[-1].upper()
            if potential_op in operators and operators[potential_op] == 1:
                lhs_str = parts[0].strip()
                op = potential_op
                parsed_successfully = True

        if not parsed_successfully:
            self.logger.error(
                f"Stage '{stage_name}': Could not parse operator from success criterion: '{criterion_str}'"
            )
            return False

        lhs_value = await self._resolve_value_for_criterion(lhs_str, stage_outputs, shared_context_for_stage)

        if lhs_value == _SENTINEL and op not in ["EXISTS", "IS_NONE"]:
            self.logger.warning(
                f"Stage '{stage_name}': LHS path '{lhs_str}' in criterion '{criterion_str}' resolved to _SENTINEL (not found)."
            )
            if op == "EXISTS":
                 self.logger.debug(f"Criterion '{criterion_str}' (EXISTS) is False because LHS path '{lhs_str}' was not found.")
                 return False
            if op == "IS_NONE":
                self.logger.debug(f"Criterion '{criterion_str}' (IS_NONE) is True because LHS path '{lhs_str}' was not found.")
                return True
            return False

        if op == "EXISTS":
            result = lhs_value != _SENTINEL
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value if lhs_value != _SENTINEL else 'NotFound'}) EXISTS -> {result}")
            return result
        elif op == "IS_NOT_EMPTY":
            if lhs_value == _SENTINEL or lhs_value is None: result = False
            elif isinstance(lhs_value, (str, list, dict, tuple, set)): result = bool(lhs_value)
            else: result = True 
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value}) IS_NOT_EMPTY -> {result}")
            return result
        elif op == "IS_EMPTY": 
            if lhs_value == _SENTINEL or lhs_value is None: result = True
            elif isinstance(lhs_value, (str, list, dict, tuple, set)): result = not bool(lhs_value)
            else: result = False 
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value}) IS_EMPTY -> {result}")
            return result
        elif op == "IS_TRUE": 
            result = lhs_value is True or (isinstance(lhs_value, str) and lhs_value.lower() == 'true')
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value}) IS_TRUE -> {result}")
            return result
        elif op == "IS_FALSE": 
            result = lhs_value is False or (isinstance(lhs_value, str) and lhs_value.lower() == 'false')
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value}) IS_FALSE -> {result}")
            return result
        elif op == "IS_NONE": 
            result = lhs_value == _SENTINEL or lhs_value is None
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value if lhs_value != _SENTINEL else 'NotFound'}) IS_NONE -> {result}")
            return result
        elif op == "IS_NOT_NONE": 
            result = lhs_value != _SENTINEL and lhs_value is not None
            self.logger.debug(f"Criterion '{criterion_str}': {lhs_str} ({lhs_value if lhs_value != _SENTINEL else 'NotFound'}) IS_NOT_NONE -> {result}")
            return result

        if rhs_str is None : 
            self.logger.error(f"Stage '{stage_name}': RHS is None for binary operator '{op}' in criterion '{criterion_str}'. This indicates a parsing logic error.")
            return False

        rhs_value: Any
        if isinstance(lhs_value, bool):
            rhs_value = rhs_str.lower() == "true"
        elif isinstance(lhs_value, (int, float)):
            try:
                rhs_value = type(lhs_value)(rhs_str)
            except ValueError:
                self.logger.warning(
                    f"Stage '{stage_name}': Could not coerce RHS '{rhs_str}' to type {type(lhs_value)} for criterion '{criterion_str}'. Comparing as strings.")
                rhs_value = rhs_str 
        elif lhs_value is None and rhs_str.upper() == "NONE": 
             rhs_value = None
        else: 
            if rhs_str.upper() == "NULL" or rhs_str.upper() == "NONE":
                rhs_value = None
            elif (rhs_str.startswith("'") and rhs_str.endswith("'")) or \
                 (rhs_str.startswith('"') and rhs_str.endswith('"')): # Quoted string
                rhs_value = rhs_str[1:-1]
            else:
                if rhs_str.lower() == "true": rhs_value = True
                elif rhs_str.lower() == "false": rhs_value = False
                elif re.fullmatch(r"-?\\d+", rhs_str): rhs_value = int(rhs_str)
                elif re.fullmatch(r"-?\\d*\\.\\d+", rhs_str): rhs_value = float(rhs_str)
                else: rhs_value = rhs_str 

        self.logger.debug(
            f"Stage '{stage_name}': Criterion '{criterion_str}'. LHS Path: '{lhs_str}', LHS Value: {lhs_value} (type {type(lhs_value)}), Op: '{op}', RHS Str: '{rhs_str}', RHS Value: {rhs_value} (type {type(rhs_value)})"
        )

        try:
            if op == "==": result = (lhs_value == rhs_value)
            elif op == "!=": result = (lhs_value != rhs_value)
            elif op == ">": result = (lhs_value > rhs_value) 
            elif op == "<": result = (lhs_value < rhs_value) 
            elif op == ">=": result = (lhs_value >= rhs_value) 
            elif op == "<=": result = (lhs_value <= rhs_value) 
            elif op == "CONTAINS":
                if isinstance(lhs_value, str): result = rhs_str in lhs_value 
                elif isinstance(lhs_value, (list, tuple, set)): result = rhs_value in lhs_value
                elif isinstance(lhs_value, dict): result = rhs_value in lhs_value 
                else: result = False
            else:
                self.logger.error(f"Stage '{stage_name}': Unsupported operator '{op}' in criterion '{criterion_str}' during comparison phase.")
                return False
        except TypeError as e:
            self.logger.error(
                f"Stage '{stage_name}': TypeError during comparison for criterion '{criterion_str}' (LHS: {lhs_value}, RHS: {rhs_value}): {e}"
            )
            return False 
        
        self.logger.debug(f"Criterion '{criterion_str}' evaluation result: {result}")
        return result 