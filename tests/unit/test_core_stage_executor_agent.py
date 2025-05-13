import unittest
from unittest.mock import patch, MagicMock, mock_open, ANY
import asyncio
from pathlib import Path
import tempfile
import shutil
import yaml

from chungoid.runtime.agents.core_stage_executor import core_stage_executor_agent, CoreStageExecutorInputs
# We will be mocking ChungoidEngine and its parts, so direct import might not be strictly needed if fully mocked.
# However, for type hints or if some passthrough is tested, it might be.
# from chungoid.engine import ChungoidEngine
# from chungoid.utils.prompt_manager import PromptManager

# Dummy class for ChungoidEngine if we can't import the real one in test env easily
class MockChungoidEngine:
    def __init__(self, project_directory):
        self.project_dir = Path(project_directory)
        self.prompt_manager = MagicMock()
        # Assume server_stages_dir is a subdirectory of where PromptManager expects stages
        # For the test, we can set it directly.
        self.prompt_manager.server_stages_dir = self.project_dir / "_test_server_stages"
        (self.project_dir / "_test_server_stages").mkdir(exist_ok=True)

    def execute_mcp_tool(self, tool_name: str, tool_arguments: dict, tool_call_id: str):
        # This will be replaced by a MagicMock instance's method
        raise NotImplementedError("MockChungoidEngine.execute_mcp_tool should be mocked")


class TestCoreStageExecutorAgent(unittest.TestCase):

    def setUp(self):
        self.test_project_dir = Path(tempfile.mkdtemp(prefix="csea_test_proj_"))
        self.mock_engine_patcher = patch('chungoid.runtime.agents.core_stage_executor.ChungoidEngine', autospec=True)
        self.MockChungoidEngineClass = self.mock_engine_patcher.start()
        self.mock_engine_instance = self.MockChungoidEngineClass.return_value
        
        # Configure the mocked engine instance's prompt_manager
        self.mock_engine_instance.prompt_manager = MagicMock()
        self.server_stages_path = self.test_project_dir / "test_server_prompts" / "stages"
        self.server_stages_path.mkdir(parents=True, exist_ok=True)
        self.mock_engine_instance.prompt_manager.server_stages_dir = self.server_stages_path

    def tearDown(self):
        self.mock_engine_patcher.stop()
        shutil.rmtree(self.test_project_dir)

    def _write_stage_yaml(self, filename: str, content: dict):
        with open(self.server_stages_path / filename, 'w') as f:
            yaml.dump(content, f)

    def test_input_validation_missing_filename(self):
        context = {
            "inputs": {
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Input validation failed", result["message"]) 
        self.assertIn("stage_definition_filename", result["error_details"][0]["loc"][0])

    def test_input_validation_missing_project_root(self):
        context = {
            "inputs": {
                "stage_definition_filename": "stage0.yaml"
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Input validation failed", result["message"]) 
        self.assertIn("current_project_root", result["error_details"][0]["loc"][0])

    def test_engine_initialization_failure(self):
        self.MockChungoidEngineClass.side_effect = RuntimeError("Engine boom!")
        context = {
            "inputs": {
                "stage_definition_filename": "stage0.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Failed to initialize engine", result["message"])
        self.assertIn("Engine boom!", result["message"])

    def test_stage_yaml_not_found(self):
        context = {
            "inputs": {
                "stage_definition_filename": "non_existent_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Sub-stage YAML file not found", result["message"])

    def test_invalid_stage_yaml_content(self):
        self._write_stage_yaml("invalid_format.yaml", {"key": {"nested_key": "value"}}) # Not a valid structure for parsing actions
        # Actually, the parsing error should come from yaml.safe_load
        with open(self.server_stages_path / "invalid_yaml_text.yaml", 'w') as f:
            f.write("key: value: not valid yaml")
        
        context = {
            "inputs": {
                "stage_definition_filename": "invalid_yaml_text.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Cannot load/parse sub-stage YAML", result["message"])

    def test_no_mcp_actions_defined(self):
        stage_content = {"name": "Stage with no actions", "prompt_template": "Do something."}
        self._write_stage_yaml("no_actions_stage.yaml", stage_content)
        context = {
            "inputs": {
                "stage_definition_filename": "no_actions_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "no_actions_defined")
        self.assertIn("Contains no direct mcp_actions", result["message"])
        self.assertEqual(result["yaml_content"], stage_content)
        self.assertIn("final_context", result)

    def test_successful_mcp_action_execution(self):
        self.mock_engine_instance.execute_mcp_tool = MagicMock(return_value={"status": "tool_success", "output": "tool done"})
        
        stage_content = {
            "name": "Stage with actions",
            "mcp_actions": [
                {"tool_name": "test_tool_1", "tool_arguments": {"arg1": "val1"}},
                {"tool_name": "test_tool_2", "tool_arguments": {"arg_from_context": "context.initial_val"}}
            ]
        }
        self._write_stage_yaml("actions_stage.yaml", stage_content)
        
        initial_master_context = {
            "initial_val": "resolved_value",
            "inputs": {
                "stage_definition_filename": "actions_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }

        result = asyncio.run(core_stage_executor_agent(initial_master_context))
        
        self.assertEqual(result["status"], "success", msg=f"Agent failed with: {result.get('message')}")
        self.assertEqual(len(result["executed_mcp_results"]), 2)
        self.assertEqual(result["executed_mcp_results"][0]["status"], "tool_success")

        self.mock_engine_instance.execute_mcp_tool.assert_any_call(
            "test_tool_1", {"arg1": "val1"}, ANY
        )
        self.mock_engine_instance.execute_mcp_tool.assert_any_call(
            "test_tool_2", {"arg_from_context": "resolved_value"}, ANY
        )
        self.assertIn("test_tool_1_0", result["final_context"]["outputs"])
        self.assertIn("test_tool_2_1", result["final_context"]["outputs"])

    def test_mcp_action_execution_failure_from_tool(self):
        # First tool succeeds, second tool reports an error in its return dict
        mock_tool_results = [
            {"status": "tool_success", "output": "tool1 done"}, 
            {"status": "tool_error", "isError": True, "error": "Tool B exploded"}
        ]
        self.mock_engine_instance.execute_mcp_tool = MagicMock(side_effect=mock_tool_results)
        
        stage_content = {
            "name": "Stage with failing action",
            "mcp_actions": [
                {"tool_name": "tool_A"}, 
                {"tool_name": "tool_B"} # This one will fail
            ]
        }
        self._write_stage_yaml("failing_actions_stage.yaml", stage_content)
        
        context = {
            "inputs": {
                "stage_definition_filename": "failing_actions_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Tool tool_B reported error", result["message"])
        self.assertEqual(len(result["executed_mcp_results"]), 2) # Both attempted
        self.assertEqual(result["failed_action_index"], 1)
        self.assertEqual(result["tool_output"], mock_tool_results[1])

    def test_mcp_action_execution_exception_in_tool(self):
        # First tool succeeds, second tool raises an exception
        def side_effect_for_tool_calls(tool_name, args, tool_call_id):
            if tool_name == "tool_A":
                return {"status": "tool_success", "output": "tool1 done"} 
            elif tool_name == "tool_B":
                raise ValueError("Tool B raised exception")
            return {}
        
        self.mock_engine_instance.execute_mcp_tool = MagicMock(side_effect=side_effect_for_tool_calls)
        
        stage_content = {
            "name": "Stage with tool exception",
            "mcp_actions": [
                {"tool_name": "tool_A"}, 
                {"tool_name": "tool_B"} # This one will raise
            ]
        }
        self._write_stage_yaml("exception_actions_stage.yaml", stage_content)
        
        context = {
            "inputs": {
                "stage_definition_filename": "exception_actions_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        
        self.assertEqual(result["status"], "error")
        self.assertIn("Exception executing tool tool_B", result["message"])
        self.assertEqual(len(result["executed_mcp_results"]), 1) # Only first one succeeded
        self.assertEqual(result["failed_action_index"], 1)

    def test_context_resolution_failure_in_args(self):
        stage_content = {
            "name": "Stage with bad context ref",
            "mcp_actions": [
                {"tool_name": "test_tool", "tool_arguments": {"bad_ref": "context.non.existent.path"}}
            ]
        }
        self._write_stage_yaml("bad_context_ref_stage.yaml", stage_content)
        
        context = {
            "inputs": {
                "stage_definition_filename": "bad_context_ref_stage.yaml",
                "current_project_root": str(self.test_project_dir)
            }
        }
        result = asyncio.run(core_stage_executor_agent(context))
        self.assertEqual(result["status"], "error")
        self.assertIn("Context resolution failed for test_tool", result["message"])
        self.assertEqual(result["failed_action_index"], 0)

if __name__ == '__main__':
    unittest.main() 