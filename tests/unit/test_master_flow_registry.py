import unittest
from pathlib import Path
import tempfile
import shutil
import yaml # For creating test YAML content

from chungoid.utils.master_flow_registry import MasterFlowRegistry, MasterFlowRegistryError
from chungoid.schemas.master_flow import MasterExecutionPlan

VALID_MASTER_FLOW_YAML_CONTENT_1 = """
id: flow_one
name: Test Master Flow One
description: First test flow.
version: 1.0.0
start_stage: stage_a
stages:
  stage_a:
    agent_id: CoreStageExecutorAgent
    number: 1.0
    inputs:
      stage_definition_filename: sub_stage_a.yaml
      current_project_root: /test/project
    next_stage: stage_b
  stage_b:
    agent_id: SomeOtherAgent
    number: 2.0
    next_stage: null
"""

VALID_MASTER_FLOW_YAML_CONTENT_2 = """
id: flow_two
name: Test Master Flow Two
start_stage: start
stages:
  start:
    agent_id: CoreStageExecutorAgent
    inputs:
      stage_definition_filename: start.yaml
    next_stage: null
"""

# For testing duplicate ID warning
DUPLICATE_ID_MASTER_FLOW_YAML_CONTENT = """
id: flow_one # Same ID as VALID_MASTER_FLOW_YAML_CONTENT_1
name: Duplicate ID Flow
start_stage: first_step
stages:
  first_step:
    agent_id: DummyAgent
    next_stage: null 
"""

INVALID_YAML_CONTENT = "key: value: this is not valid yaml"

YAML_MISSING_ID = """
name: Flow Missing ID
start_stage: start
stages:
  start:
    agent_id: AgentX
"""

YAML_MISSING_START_STAGE = """
id: flow_missing_start
name: Flow Missing Start Stage
stages:
  start:
    agent_id: AgentX
"""

YAML_MISSING_STAGES = """
id: flow_missing_stages
name: Flow Missing Stages
start_stage: start
"""

class TestMasterFlowRegistry(unittest.TestCase):

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp(prefix="chungoid_test_master_flows_"))
        # print(f"Setup test directory: {self.test_dir}")

    def tearDown(self):
        shutil.rmtree(self.test_dir)
        # print(f"Teardown test directory: {self.test_dir}")

    def _write_file(self, filename: str, content: str):
        (self.test_dir / filename).write_text(content)

    def test_scan_empty_directory(self):
        registry = MasterFlowRegistry(self.test_dir)
        self.assertEqual(len(registry.list_ids()), 0)
        self.assertEqual(len(registry.list_plans()), 0)
        self.assertEqual(len(registry.get_scan_errors()), 0)

    def test_scan_non_existent_directory(self):
        non_existent_dir = self.test_dir / "does_not_exist"
        registry = MasterFlowRegistry(non_existent_dir)
        self.assertEqual(len(registry.list_ids()), 0)
        self.assertEqual(len(registry.list_plans()), 0)
        self.assertTrue(len(registry.get_scan_errors()) >= 1)
        self.assertIn(str(non_existent_dir), registry.get_scan_errors()[0])

    def test_scan_valid_flows(self):
        self._write_file("flow1.yaml", VALID_MASTER_FLOW_YAML_CONTENT_1)
        self._write_file("flow2.yaml", VALID_MASTER_FLOW_YAML_CONTENT_2)
        self._write_file("not_a_flow.txt", "some text content") # Should be ignored

        registry = MasterFlowRegistry(self.test_dir)
        self.assertEqual(len(registry.list_ids()), 2)
        self.assertEqual(len(registry.list_plans()), 2)
        self.assertEqual(len(registry.get_scan_errors()), 0)

        self.assertIn("flow_one", registry.list_ids())
        self.assertIn("flow_two", registry.list_ids())

        plan1 = registry.get("flow_one")
        self.assertIsNotNone(plan1)
        self.assertIsInstance(plan1, MasterExecutionPlan)
        self.assertEqual(plan1.id, "flow_one")
        self.assertEqual(plan1.name, "Test Master Flow One")
        self.assertTrue(hasattr(plan1, '_source_file'))
        self.assertEqual(Path(plan1._source_file).name, "flow1.yaml")

        plan2 = registry.get("flow_two")
        self.assertIsNotNone(plan2)
        self.assertEqual(plan2.id, "flow_two")

    def test_scan_with_invalid_yaml(self):
        self._write_file("valid.yaml", VALID_MASTER_FLOW_YAML_CONTENT_1)
        self._write_file("invalid.yaml", INVALID_YAML_CONTENT)
        registry = MasterFlowRegistry(self.test_dir)
        
        self.assertEqual(len(registry.list_ids()), 1) # Only valid one loaded
        self.assertIn("flow_one", registry.list_ids())
        self.assertEqual(len(registry.get_scan_errors()), 1)
        self.assertIn("invalid.yaml", registry.get_scan_errors()[0])
        self.assertIn("Failed to load or parse", registry.get_scan_errors()[0])

    def test_scan_with_missing_required_fields(self):
        self._write_file("missing_id.yaml", YAML_MISSING_ID)
        self._write_file("missing_start.yaml", YAML_MISSING_START_STAGE)
        self._write_file("missing_stages.yaml", YAML_MISSING_STAGES)
        self._write_file("valid_flow.yaml", VALID_MASTER_FLOW_YAML_CONTENT_1)

        registry = MasterFlowRegistry(self.test_dir)
        self.assertEqual(len(registry.list_ids()), 1) # Only valid_flow should load
        self.assertIn("flow_one", registry.list_ids())
        self.assertEqual(len(registry.get_scan_errors()), 3)

        errors_str = "\n".join(registry.get_scan_errors())
        self.assertIn("missing_id.yaml", errors_str)
        self.assertIn("missing_start.yaml", errors_str)
        self.assertIn("missing_stages.yaml", errors_str)
        # Pydantic's from_yaml raises ValueError for missing fields
        self.assertIn("missing required 'id', 'start_stage', or 'stages' key", errors_str)

    def test_scan_with_duplicate_ids(self):
        self._write_file("flow_A.yaml", VALID_MASTER_FLOW_YAML_CONTENT_1) # id: flow_one
        # Rename second file to ensure it's processed last alphabetically by glob
        self._write_file("flow_Z_duplicate.yaml", DUPLICATE_ID_MASTER_FLOW_YAML_CONTENT) # id: flow_one

        registry = MasterFlowRegistry(self.test_dir)
        # Only one plan with id 'flow_one' should be in the registry (the last one scanned)
        self.assertEqual(len(registry.list_ids()), 1)
        self.assertIn("flow_one", registry.list_ids())
        plan = registry.get("flow_one")
        self.assertIsNotNone(plan)
        # Now expect the name from the Z file
        self.assertEqual(plan.name, "Duplicate ID Flow") 

        # One scan error should be recorded for the duplicate ID
        self.assertEqual(len(registry.get_scan_errors()), 1)
        self.assertIn("Duplicate ID 'flow_one'", registry.get_scan_errors()[0])
        # The error message should mention the file causing the overwrite (the Z file)
        self.assertIn("flow_Z_duplicate.yaml", registry.get_scan_errors()[0])

    def test_get_non_existent_flow(self):
        registry = MasterFlowRegistry(self.test_dir)
        self.assertIsNone(registry.get("non_existent_id"))

    def test_rescan_functionality(self):
        registry = MasterFlowRegistry(self.test_dir)
        self.assertEqual(len(registry.list_ids()), 0)

        self._write_file("flow_rescan.yaml", VALID_MASTER_FLOW_YAML_CONTENT_1)
        registry.rescan()
        self.assertEqual(len(registry.list_ids()), 1)
        self.assertIn("flow_one", registry.list_ids())
        self.assertEqual(len(registry.get_scan_errors()), 0)

        # Add another file and rescan
        self._write_file("flow_rescan_2.yaml", VALID_MASTER_FLOW_YAML_CONTENT_2)
        registry.rescan()
        self.assertEqual(len(registry.list_ids()), 2)
        self.assertIn("flow_one", registry.list_ids())
        self.assertIn("flow_two", registry.list_ids())
        self.assertEqual(len(registry.get_scan_errors()), 0)

        # Remove a file and rescan
        (self.test_dir / "flow_rescan.yaml").unlink()
        registry.rescan()
        self.assertEqual(len(registry.list_ids()), 1)
        self.assertNotIn("flow_one", registry.list_ids())
        self.assertIn("flow_two", registry.list_ids())
        self.assertEqual(len(registry.get_scan_errors()), 0)

if __name__ == '__main__':
    unittest.main() 