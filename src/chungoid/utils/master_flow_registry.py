from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, List

from chungoid.schemas.master_flow import MasterExecutionPlan # Import the schema

logger = logging.getLogger(__name__)

class MasterFlowRegistryError(Exception):
    """Custom exception for Master Flow Registry errors."""
    pass

class MasterFlowRegistry:
    """Manages discovery and loading of Master Execution Plans from YAML files."""

    def __init__(self, master_flows_dir: str | Path):
        """
        Initializes the registry and scans the specified directory for Master Flow YAML files.

        Args:
            master_flows_dir: Path to the directory containing master_flow_*.yaml files.
        """
        self.master_flows_dir = Path(master_flows_dir).resolve()
        self._plans: Dict[str, MasterExecutionPlan] = {}
        self._scan_errors: List[str] = []
        self._scan_flows()

    def _scan_flows(self) -> None:
        """Scans the directory for Master Flow YAML files and loads them."""
        self._plans = {}
        self._scan_errors = []
        if not self.master_flows_dir.is_dir():
            msg = f"Master flows directory not found or not a directory: {self.master_flows_dir}"
            logger.warning(msg)
            self._scan_errors.append(msg)
            # Don't raise here, allow registry to exist but be empty/report errors
            return

        logger.info(f"Scanning for Master Flows in: {self.master_flows_dir}")
        flow_count = 0
        # Sort the globbed files by name to ensure deterministic processing order
        try:
            sorted_yaml_files = sorted(list(self.master_flows_dir.glob("*.yaml")), key=lambda p: p.name)
        except Exception as glob_err:
            logger.error(f"Error during globbing or sorting in {self.master_flows_dir}: {glob_err}")
            self._scan_errors.append(f"Error accessing directory contents: {glob_err}")
            return

        for yaml_file in sorted_yaml_files: # Iterate over sorted list
            # No need for yaml_file.is_file() check as glob should only return files matching pattern
            logger.debug(f"Attempting to load Master Flow from: {yaml_file.name}")
            try:
                with open(yaml_file, 'r') as f:
                    yaml_text = f.read()
                    plan = MasterExecutionPlan.from_yaml(yaml_text)
                
                if plan.id in self._plans:
                    logger.warning(f"Duplicate Master Flow ID '{plan.id}' found in '{yaml_file.name}'. Overwriting previous definition from {self._plans[plan.id]._source_file if hasattr(self._plans[plan.id], '_source_file') else 'unknown file'}.")
                    self._scan_errors.append(f"Duplicate ID '{plan.id}' in {yaml_file.name}")
                
                # Store the source file path on the plan object for reference
                plan._source_file = str(yaml_file) 
                self._plans[plan.id] = plan
                flow_count += 1
                logger.debug(f"Successfully loaded Master Flow ID '{plan.id}' from {yaml_file.name}")

            except Exception as e:
                msg = f"Failed to load or parse Master Flow from {yaml_file.name}: {e}"
                logger.error(msg)
                self._scan_errors.append(msg)
        
        logger.info(f"Master Flow scan complete. Loaded {flow_count} plans. Found {len(self._scan_errors)} errors.")

    def get(self, flow_id: str) -> Optional[MasterExecutionPlan]:
        """Retrieves a loaded MasterExecutionPlan by its ID."""
        return self._plans.get(flow_id)

    def list_ids(self) -> List[str]:
        """Returns a list of IDs of all successfully loaded Master Flows."""
        return list(self._plans.keys())

    def list_plans(self) -> List[MasterExecutionPlan]:
        """Returns a list of all successfully loaded MasterExecutionPlan objects."""
        return list(self._plans.values())

    def get_scan_errors(self) -> List[str]:
        """Returns a list of errors encountered during the last scan."""
        return self._scan_errors

    def rescan(self) -> None:
        """Clears the current registry and rescans the directory."""
        logger.info(f"Rescanning Master Flows directory: {self.master_flows_dir}")
        self._scan_flows()

# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # Create dummy directory and files for testing
    temp_dir = Path("./temp_master_flows_test")
    temp_dir.mkdir(exist_ok=True)
    
    # Valid flow
    valid_flow_yaml = """
id: valid_flow_1
name: My Valid Master Flow
description: A test flow.
start_stage: start
stages:
  start:
    agent_id: CoreStageExecutorAgent
    inputs:
      stage_definition_filename: stage0.yaml
      current_project_root: /path/to/project
    next_stage: null
"""
    (temp_dir / "valid1.yaml").write_text(valid_flow_yaml)

    # Flow with duplicate ID
    duplicate_flow_yaml = """
id: valid_flow_1 # Same ID!
name: Another Flow (duplicate ID)
description: This should cause a warning.
start_stage: step_a
stages:
  step_a:
    agent_id: SomeOtherAgent
    next_stage: null
"""
    (temp_dir / "duplicate.yaml").write_text(duplicate_flow_yaml)

    # Invalid YAML
    invalid_yaml = "key: value: nested invalid" 
    (temp_dir / "invalid.yaml").write_text(invalid_yaml)

    # File missing required field (id)
    missing_id_yaml = """
name: Missing ID Flow
start_stage: start
stages: {}
"""
    (temp_dir / "missing_id.yaml").write_text(missing_id_yaml)

    print(f"\n--- Initializing MasterFlowRegistry for {temp_dir} ---")
    registry = MasterFlowRegistry(temp_dir)

    print("\n--- Listing Loaded Flow IDs ---")
    print(registry.list_ids())

    print("\n--- Getting a Specific Flow ---")
    plan = registry.get("valid_flow_1")
    if plan:
        print(f"Got plan '{plan.id}', Name: '{plan.name}', Source: {getattr(plan, '_source_file', 'N/A')}")
    else:
        print("Plan valid_flow_1 not found (This shouldn't happen in this example)")

    print("\n--- Scan Errors ---")
    errors = registry.get_scan_errors()
    if errors:
        for err in errors:
            print(f" - {err}")
    else:
        print("(No scan errors)") # Should see errors for duplicate, invalid, missing id
    
    # Cleanup dummy files
    import shutil
    # Uncomment below to clean up
    # try:
    #     shutil.rmtree(temp_dir)
    #     print(f"\nCleaned up {temp_dir}")
    # except OSError as e:
    #     print(f"\nError removing {temp_dir}: {e}") 