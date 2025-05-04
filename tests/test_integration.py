import unittest
import os
import shutil
from pathlib import Path
import sys
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import chromadb
import logging
import json

# Add project root to sys.path to allow importing project modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now import necessary components AFTER adjusting sys.path
# Assume server functions are refactored or accessible for testing
# This might require adjustments in chungoidmcp.py or importing specific handlers
from chungoidmcp import (
    handle_initialize_project,
    handle_get_project_status,
    handle_execute_next_stage,
    handle_submit_stage_artifacts,
    handle_load_reflections,
    handle_retrieve_reflections_tool,
    handle_get_file,
    # We'll need a way to simulate the context or pass necessary args
)
from utils.state_manager import StateManager, StatusFileError  # Import error class
from utils.chroma_utils import ChromaOperationError # <<< IMPORT FROM CORRECT MODULE
from utils.exceptions import StageExecutionError, ToolExecutionError # Import other exceptions


class TestIntegration(unittest.TestCase):
    TEST_DIR = Path("./test_project_integration")
    CHUNGOID_DIR = TEST_DIR / ".chungoid"
    STATUS_FILE = CHUNGOID_DIR / "project_status.json"
    logger = logging.getLogger(__name__)

    def setUp(self):
        # Create a clean test directory for each test
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)
        self.TEST_DIR.mkdir()
        # We need to simulate the server context or modify handlers to accept paths
        # For now, let's assume handlers can accept target_directory directly

    def tearDown(self):
        # Clean up the test directory after each test
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)

    def test_01_initialize_and_get_status(self):
        """Test initializing a project and then getting its status."""
        print(f"\nRunning test: test_01_initialize_and_get_status in {os.getcwd()}")

        # Define an async inner function to run the async handlers
        async def run_async_test():
            # Initialize
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            # Call with target_directory and ctx=None
            init_result = await handle_initialize_project(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Initialization result: {init_result}")
            self.assertTrue(
                self.CHUNGOID_DIR.exists(), msg="Chungoid directory should exist after init"
            )
            self.assertTrue(self.STATUS_FILE.exists(), msg="Status file should exist after init")
            self.assertIn(
                "Project initialized successfully",
                init_result.get("message", ""),
                msg="Init success message expected",
            )

            # Get Status - Pass target_directory for test isolation
            print(f"Getting project status from: {self.TEST_DIR.resolve()}")
            status_result = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Status result: {status_result}")
            self.assertEqual(status_result.get("status"), "success", msg="Getting status failed")
            status_data = status_result.get("data", {})
            self.assertIsInstance(status_data, dict, msg="Status data should be a dict")
            self.assertEqual(
                status_data.get("project_initialized"), True, msg="Project should be initialized"
            )
            # FIX: Check the stage within the last_status dict
            last_status_info = status_data.get("last_status", {})
            self.assertEqual(last_status_info.get("stage"), 0, msg="Initial stage should be 0")
            self.assertEqual(
                last_status_info.get("status"), "PENDING", msg="Initial status should be PENDING"
            )
            self.assertIsInstance(
                status_data.get("full_history"), list, msg="History should be a list"
            )

        # Run the async function using asyncio.run()
        asyncio.run(run_async_test())

    # Add more integration tests here...
    def test_02_execute_and_submit_stage_0(self):
        """Test executing stage 0 and submitting artifacts."""
        print(f"\nRunning test: test_02_execute_and_submit_stage_0 in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize Project ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            init_result = await handle_initialize_project(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(init_result.get("status"), "success", msg="Initialization failed")
            print("Project initialized.")

            # --- Execute Stage 0 - Pass target_directory ---
            print("Executing next stage (should be Stage 0)...")
            exec_result = await handle_execute_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Execution result: {exec_result}")
            self.assertEqual(exec_result.get("status"), "success", msg="Execute stage failed")
            # FIX: Check prompt is directly in the result dict
            self.assertIn(
                "STAGE 0 BEGIN", exec_result.get("prompt", ""), msg="Stage 0 prompt expected"
            )
            self.assertEqual(exec_result.get("next_stage"), 0.0, msg="Next stage should be 0.0")
            print("Stage 0 executed successfully.")

            # --- Submit Stage 0 Artifacts - Pass target_directory ---
            print("Submitting artifacts for Stage 0...")
            artifacts = {"dev-docs/stage0_output.txt": "This is a dummy output for Stage 0."}
            submit_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=0,
                generated_artifacts=artifacts,
                stage_result_status="PASS",
                ctx=None,
            )
            print(f"Submission result: {submit_result}")
            self.assertEqual(submit_result.get("status"), "success", msg="Submit artifacts failed")
            self.assertIn(
                "dev-docs/stage0_output.txt",
                submit_result.get("written_files", []),
                msg="Artifact path missing from result",
            )
            print("Stage 0 artifacts submitted successfully.")

            # <<< ADD DEBUG: Read file directly after submit >>>
            status_file_path = self.TEST_DIR / ".chungoid" / "project_status.json"
            if status_file_path.exists():
                with open(status_file_path, "r") as f:
                    print(f"DEBUG: Content of {status_file_path} AFTER submit:\n{f.read()}")
            else:
                print(f"DEBUG: {status_file_path} does not exist AFTER submit.")
            # <<< END DEBUG >>>

            # --- Verify Status Update - Pass target_directory ---
            print("Verifying status update...")
            status_result = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(
                status_result.get("status"), "success", msg="Getting status after submit failed"
            )
            status_data_summary = status_result.get("data", {})
            self.assertIsInstance(
                status_data_summary, dict, msg="Status data summary should be a dict"
            )

            # FIX: Check the stage within the last_status dict
            last_status_info_s0 = status_data_summary.get("last_status", {})
            self.assertEqual(
                last_status_info_s0.get("stage"), 0, msg="Stage after submit should be 0"
            )
            self.assertEqual(
                last_status_info_s0.get("status"), "PASS", msg="Status after submit should be PASS"
            )
            self.assertIn(
                list(artifacts.keys())[0],
                last_status_info_s0.get("artifacts", []),
                msg="Artifact missing from status",
            )
            print("Status correctly updated to PASS for Stage 0.")

        # Run the async part
        asyncio.run(run_async_test())

    def test_03_execute_and_submit_stage_1(self):
        """Test executing stage 1 and submitting artifacts."""
        print(f"\nRunning test: test_03_execute_and_submit_stage_1 in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize and Complete Stage 0 ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print("Project initialized.")

            # Submit Stage 0 to advance state - Pass target_directory
            artifacts_s0 = {"dev-docs/stage0_output.txt": "Dummy output"}
            submit_s0_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=0,
                generated_artifacts=artifacts_s0,
                stage_result_status="PASS",  # Assuming PASS allows proceeding
                ctx=None,
            )
            self.assertEqual(
                submit_s0_result.get("status"), "success", msg="Submit Stage 0 artifacts failed"
            )
            print("Stage 0 submitted.")

            # --- Execute Stage 1 - Pass target_directory ---
            print("Executing next stage (should be Stage 1)...")
            exec_result_s1 = await handle_execute_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Execution result S1: {exec_result_s1}")
            # Check if execution itself was successful
            self.assertEqual(
                exec_result_s1.get("status"), "success", msg="Execute Stage 1 call failed"
            )
            # Check the stage number
            self.assertEqual(exec_result_s1.get("next_stage"), 1.0, msg="Next stage should be 1.0")
            # Check the prompt content
            self.assertIn(
                "STAGE 1 BEGIN", exec_result_s1.get("prompt", ""), msg="Stage 1 prompt expected"
            )
            print("Stage 1 executed successfully.")

            # --- Submit Stage 1 Artifacts - Pass target_directory ---
            print("Submitting artifacts for Stage 1...")
            artifacts_s1 = {
                "dev-docs/design/validation_report.json": '{"stage": 1, "status": "PASS"}'
            }
            submit_s1_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=1,
                generated_artifacts=artifacts_s1,
                stage_result_status="PASS",
                ctx=None,
            )
            print(f"Submission result S1: {submit_s1_result}")
            self.assertEqual(
                submit_s1_result.get("status"), "success", msg="Submit Stage 1 artifacts failed"
            )
            self.assertIn(
                "dev-docs/design/validation_report.json",
                submit_s1_result.get("written_files", []),
                msg="Stage 1 artifact path missing from result",
            )
            print("Stage 1 artifacts submitted successfully.")

            # --- Verify Status Update for Stage 1 - Pass target_directory ---
            print("Verifying status update for Stage 1...")
            status_result_s1 = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(
                status_result_s1.get("status"),
                "success",
                msg="Getting status after Stage 1 submit failed",
            )
            status_data_summary_s1 = status_result_s1.get("data", {})
            self.assertIsInstance(
                status_data_summary_s1, dict, msg="Status data summary should be a dict"
            )

            # The *highest completed* stage should now be 1, but current might be pending for 2
            # Let's check the history for the specific Stage 1 entry
            full_history_s1 = status_data_summary_s1.get("full_history", [])
            self.assertIsInstance(full_history_s1, list)
            stage1_status = next(
                (s for s in reversed(full_history_s1) if s.get("stage") == 1), None
            )
            self.assertIsNotNone(stage1_status, msg="Stage 1 status entry not found in history")
            self.assertEqual(
                stage1_status.get("status"), "PASS", msg="Stage 1 status should be PASS"
            )
            self.assertIn(
                "dev-docs/design/validation_report.json",
                stage1_status.get("artifacts", []),
                msg="Artifact missing from Stage 1 status entry",
            )
            print("Status correctly updated to PASS for Stage 1.")

        # Run the async part
        asyncio.run(run_async_test())

    def test_04_execute_and_submit_stage_2(self):
        """Test executing stage 2 and submitting artifacts."""
        print(f"\nRunning test: test_04_execute_and_submit_stage_2 in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize and Complete Stages 0 & 1 ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            # Ensure this call has target_directory
            await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print("Project initialized.")

            # Submit Stages 0, 1 - Pass target_directory
            for i in range(2):
                print(f"Submitting Stage {i}...")
                artifacts = {f"dev-docs/stage{i}_output.dummy": f"Dummy output {i}"}
                # Create dummy files/dirs if needed by submit handler's logic
                if i == 1:
                    artifacts = {"dev-docs/design/validation_report.json": '{"status": "PASS"}'}
                # Ensure directories exist if submit handler creates files
                for artifact_path in artifacts:
                    (self.TEST_DIR / Path(artifact_path).parent).mkdir(parents=True, exist_ok=True)
                    if isinstance(artifacts[artifact_path], str):
                        (self.TEST_DIR / Path(artifact_path)).write_text(artifacts[artifact_path])

                await handle_submit_stage_artifacts(
                    target_directory=str(self.TEST_DIR.resolve()),
                    stage_number=i,
                    generated_artifacts=artifacts,
                    stage_result_status="PASS",
                    ctx=None,  # Pass ctx=None for test
                )
                print(f"Stage {i} submitted.")

            # --- Execute Stage 2 - Pass target_directory ---
            print("Executing next stage (should be Stage 2)...")
            exec_result_s2 = await handle_execute_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Execution result S2: {exec_result_s2}")
            self.assertEqual(exec_result_s2.get("status"), "success", "Execute Stage 2 call failed")
            self.assertEqual(exec_result_s2.get("next_stage"), 2.0, "Next stage should be 2.0")
            self.assertIn(
                "STAGE 2 BEGIN", exec_result_s2.get("prompt", ""), "Stage 2 prompt expected"
            )
            print("Stage 2 executed successfully.")

            # --- Submit Stage 2 Artifacts - Pass target_directory ---
            print("Submitting artifacts for Stage 2...")
            artifacts_s2 = {
                "dev-docs/planning/implementation_plan.md": "Plan details...",
                "dev-docs/planning/detailed_interfaces.md": "Interface details...",
            }
            submit_s2_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=2,
                generated_artifacts=artifacts_s2,
                stage_result_status="PASS",
                ctx=None,
            )
            print(f"Submission result S2: {submit_s2_result}")
            self.assertEqual(
                submit_s2_result.get("status"), "success", "Submit Stage 2 artifacts failed"
            )
            self.assertIn(
                "dev-docs/planning/implementation_plan.md",
                submit_s2_result.get("written_files", []),
                "Stage 2 artifact 1 missing",
            )
            self.assertIn(
                "dev-docs/planning/detailed_interfaces.md",
                submit_s2_result.get("written_files", []),
                "Stage 2 artifact 2 missing",
            )
            print("Stage 2 artifacts submitted successfully.")

            # --- Verify Status Update for Stage 2 - Pass target_directory ---
            print("Verifying status update for Stage 2...")
            status_result_s2 = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(
                status_result_s2.get("status"),
                "success",
                "Getting status after Stage 2 submit failed",
            )
            status_data_summary_s2 = status_result_s2.get("data", {})
            full_history_s2 = status_data_summary_s2.get("full_history", [])
            stage2_status = next(
                (s for s in reversed(full_history_s2) if s.get("stage") == 2), None
            )
            self.assertIsNotNone(stage2_status, "Stage 2 status entry not found")
            self.assertEqual(stage2_status.get("status"), "PASS", "Stage 2 status should be PASS")
            self.assertIn(
                "dev-docs/planning/implementation_plan.md", stage2_status.get("artifacts", [])
            )
            self.assertIn(
                "dev-docs/planning/detailed_interfaces.md", stage2_status.get("artifacts", [])
            )
            print("Status correctly updated to PASS for Stage 2.")

        # Run the async part
        asyncio.run(run_async_test())

    def test_05_execute_and_submit_stage_3(self):
        """Test executing stage 3 and submitting artifacts."""
        print(f"\nRunning test: test_05_execute_and_submit_stage_3 in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize and Complete Stages 0, 1 & 2 ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print("Project initialized.")

            # Submit Stages 0, 1, 2
            for i in range(3):
                print(f"Submitting Stage {i}...")
                artifacts = {f"dev-docs/stage{i}_output.dummy": f"Dummy output {i}"}
                # Create dummy files/dirs if needed by submit handler's logic
                if i == 1:
                    artifacts = {"dev-docs/design/validation_report.json": '{"status": "PASS"}'}
                if i == 2:
                    artifacts = {
                        "dev-docs/planning/implementation_plan.md": "Plan",
                        "dev-docs/planning/detailed_interfaces.md": "Interfaces",
                    }
                # Ensure directories exist if submit handler creates files
                for artifact_path in artifacts:
                    (self.TEST_DIR / Path(artifact_path).parent).mkdir(parents=True, exist_ok=True)
                    if isinstance(artifacts[artifact_path], str):
                        (self.TEST_DIR / Path(artifact_path)).write_text(artifacts[artifact_path])

                await handle_submit_stage_artifacts(
                    target_directory=str(self.TEST_DIR.resolve()),
                    stage_number=i,
                    generated_artifacts=artifacts,
                    stage_result_status="PASS",
                )
                print(f"Stage {i} submitted.")

            # --- Execute Stage 3 - Pass target_directory ---
            print("Executing next stage (should be Stage 3)...")
            exec_result_s3 = await handle_execute_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Execution result S3: {exec_result_s3}")
            self.assertEqual(exec_result_s3.get("status"), "success", "Execute Stage 3 call failed")
            self.assertEqual(exec_result_s3.get("next_stage"), 3.0, "Next stage should be 3.0")
            self.assertIn(
                "STAGE 3 BEGIN", exec_result_s3.get("prompt", ""), "Stage 3 prompt expected"
            )
            print("Stage 3 executed successfully.")

            # --- Submit Stage 3 Artifacts - Pass target_directory ---
            # Note: Stage 3 primarily creates code/tests. We submit report artifacts.
            print("Submitting artifacts for Stage 3...")
            artifacts_s3 = {
                "dev-docs/reports/static_analysis_report.json": '{"ruff": "PASS", "bandit": "PASS"}',
                "dev-docs/reports/unit_test_report.json": '{"status": "PASS", "coverage": "X%"}',
            }
            # Create dummy report files so submission doesn't fail file existence check
            report_dir = self.TEST_DIR / "dev-docs" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            for report_path, content in artifacts_s3.items():
                (self.TEST_DIR / report_path).write_text(content)

            submit_s3_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=3,
                generated_artifacts=artifacts_s3,  # Pass the dict, handler writes content
                stage_result_status="PASS",
                ctx=None,
            )
            print(f"Submission result S3: {submit_s3_result}")
            self.assertEqual(
                submit_s3_result.get("status"), "success", "Submit Stage 3 artifacts failed"
            )
            # Check expected written files (relative to TEST_DIR)
            self.assertIn(
                "dev-docs/reports/static_analysis_report.json",
                submit_s3_result.get("written_files", []),
                "Stage 3 artifact 1 missing",
            )
            self.assertIn(
                "dev-docs/reports/unit_test_report.json",
                submit_s3_result.get("written_files", []),
                "Stage 3 artifact 2 missing",
            )
            print("Stage 3 artifacts submitted successfully.")

            # --- Verify Status Update for Stage 3 - Pass target_directory ---
            print("Verifying status update for Stage 3...")
            status_result_s3 = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(
                status_result_s3.get("status"),
                "success",
                "Getting status after Stage 3 submit failed",
            )
            status_data_summary_s3 = status_result_s3.get("data", {})
            full_history_s3 = status_data_summary_s3.get("full_history", [])
            stage3_status = next(
                (s for s in reversed(full_history_s3) if s.get("stage") == 3), None
            )
            self.assertIsNotNone(stage3_status, "Stage 3 status entry not found")
            self.assertEqual(stage3_status.get("status"), "PASS", "Stage 3 status should be PASS")
            self.assertIn(
                "dev-docs/reports/static_analysis_report.json", stage3_status.get("artifacts", [])
            )
            self.assertIn(
                "dev-docs/reports/unit_test_report.json", stage3_status.get("artifacts", [])
            )
            print("Status correctly updated to PASS for Stage 3.")

        # Run the async part
        asyncio.run(run_async_test())

    def test_06_execute_and_submit_stage_4(self):
        """Test executing stage 4 and submitting artifacts."""
        print(f"\nRunning test: test_06_execute_and_submit_stage_4 in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize and Complete Stages 0, 1, 2 & 3 (Init sets context) ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print("Project initialized.")

            # Submit Stages 0, 1, 2
            for i in range(3):
                print(f"Submitting Stage {i}...")
                artifacts = {f"dev-docs/stage{i}_output.dummy": f"Dummy output {i}"}
                # Create dummy files/dirs if needed by submit handler's logic
                if i == 1:
                    artifacts = {"dev-docs/design/validation_report.json": '{"status": "PASS"}'}
                if i == 2:
                    artifacts = {
                        "dev-docs/planning/implementation_plan.md": "Plan",
                        "dev-docs/planning/detailed_interfaces.md": "Interfaces",
                    }
                # Ensure directories exist if submit handler creates files
                for artifact_path in artifacts:
                    (self.TEST_DIR / Path(artifact_path).parent).mkdir(parents=True, exist_ok=True)
                    if isinstance(artifacts[artifact_path], str):
                        (self.TEST_DIR / Path(artifact_path)).write_text(artifacts[artifact_path])

                await handle_submit_stage_artifacts(
                    target_directory=str(self.TEST_DIR.resolve()),
                    stage_number=i,
                    generated_artifacts=artifacts,
                    stage_result_status="PASS",
                )
                print(f"Stage {i} submitted.")

            # Submit Stage 3 (reports)
            print("Submitting Stage 3...")
            artifacts_s3 = {
                "dev-docs/reports/static_analysis_report.json": '{"ruff": "PASS"}',
                "dev-docs/reports/unit_test_report.json": '{"status": "PASS"}',
            }
            report_dir = self.TEST_DIR / "dev-docs" / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            for report_path, content in artifacts_s3.items():
                (self.TEST_DIR / report_path).write_text(content)
            await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=3,
                generated_artifacts=artifacts_s3,
                stage_result_status="PASS",
            )
            print("Stage 3 submitted.")

            # --- Execute Stage 4 - Pass target_directory ---
            print("Executing next stage (should be Stage 4)...")
            exec_result_s4 = await handle_execute_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Execution result S4: {exec_result_s4}")
            self.assertEqual(exec_result_s4.get("status"), "success", "Execute Stage 4 call failed")
            self.assertEqual(exec_result_s4.get("next_stage"), 4.0, "Next stage should be 4.0")
            self.assertIn(
                "STAGE 4 BEGIN", exec_result_s4.get("prompt", ""), "Stage 4 prompt expected"
            )
            print("Stage 4 executed successfully.")

            # --- Submit Stage 4 Artifacts - Pass target_directory ---
            print("Submitting artifacts for Stage 4...")
            artifacts_s4 = {
                "dev-docs/validation/integration_report.json": '{"status": "PASS"}',
                "dev-docs/validation/security_report.json": '{"status": "PASS"}',
                # Add performance report if generated by stage 4 normally
                # "dev-docs/validation/performance_report.json": '{"status": "PASS"}'
            }
            # Create dummy report files
            validation_dir = self.TEST_DIR / "dev-docs" / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            for report_path, content in artifacts_s4.items():
                (self.TEST_DIR / report_path).write_text(content)

            submit_s4_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=4,
                generated_artifacts=artifacts_s4,
                stage_result_status="PASS",
            )
            print(f"Submission result S4: {submit_s4_result}")
            self.assertEqual(
                submit_s4_result.get("status"), "success", "Submit Stage 4 artifacts failed"
            )
            self.assertIn(
                "dev-docs/validation/integration_report.json",
                submit_s4_result.get("written_files", []),
            )
            self.assertIn(
                "dev-docs/validation/security_report.json",
                submit_s4_result.get("written_files", []),
            )
            print("Stage 4 artifacts submitted successfully.")

            # --- Verify Status Update for Stage 4 - Pass target_directory ---
            print("Verifying status update for Stage 4...")
            status_result_s4 = await handle_get_project_status(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(
                status_result_s4.get("status"),
                "success",
                "Getting status after Stage 4 submit failed",
            )
            status_data_summary_s4 = status_result_s4.get("data", {})
            full_history_s4 = status_data_summary_s4.get("full_history", [])
            stage4_status = next(
                (s for s in reversed(full_history_s4) if s.get("stage") == 4), None
            )
            self.assertIsNotNone(stage4_status, "Stage 4 status entry not found")
            self.assertEqual(stage4_status.get("status"), "PASS", "Stage 4 status should be PASS")
            self.assertIn(
                "dev-docs/validation/integration_report.json", stage4_status.get("artifacts", [])
            )
            self.assertIn(
                "dev-docs/validation/security_report.json", stage4_status.get("artifacts", [])
            )
            print("Status correctly updated to PASS for Stage 4.")

        # Run the async part
        asyncio.run(run_async_test())

    # Test ChromaDB interactions (add/get context/artifacts)
    def test_07_chromadb_operations(self):
        """Test direct interaction with ChromaDB via utils (assuming sync utils)."""
        # NOTE: This test assumes chroma_utils functions are synchronous or handled appropriately.
        # If chroma_utils functions are async, this test needs adjustment (e.g., run in async loop).
        print(f"\nRunning test: test_07_chromadb_operations in {os.getcwd()}")

        # We need a real or mock ChromaDB client accessible for this.
        # Let's bypass the singleton getter and directly instantiate for isolation,
        # perhaps using a temporary persistent client.
        temp_chroma_dir = self.TEST_DIR / "temp_chroma"
        try:
            shutil.rmtree(temp_chroma_dir, ignore_errors=True)
            test_client = chromadb.PersistentClient(path=str(temp_chroma_dir))
            self.assertIsNotNone(test_client, "Failed to create temporary PersistentClient")

            collection_name = "test_integration_coll"
            # Mock get_chroma_client to return our test_client instance
            with patch('utils.chroma_utils.get_chroma_client', return_value=test_client):

                # Test get_or_create_collection
                # <<< Assuming get_or_create_collection itself is now ASYNC >>>
                async def run_create():
                    collection = await chroma_utils.get_or_create_collection(collection_name)
                    self.assertIsNotNone(collection, "Failed to get/create collection")
                    self.assertEqual(collection.name, collection_name)
                asyncio.run(run_create())

                # Test add_documents (assuming it's async)
                docs = ["doc1", "doc2"]
                ids = ["id1", "id2"]
                metadatas = [{"type": "test"}, {"type": "test"}]
                async def run_add():
                    success = await chroma_utils.add_documents(collection_name, docs, metadatas, ids)
                    self.assertTrue(success, "Failed to add documents")
                asyncio.run(run_add())

                # Test query_documents (assuming it's async)
                async def run_query():
                    results = await chroma_utils.query_documents(collection_name, query_texts=["doc1"], n_results=1)
                    self.assertIsNotNone(results)
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0]["id"], "id1")
                asyncio.run(run_query())

        finally:
            # Clean up temp ChromaDB directory
            shutil.rmtree(temp_chroma_dir, ignore_errors=True)
            print("Cleaned up temp ChromaDB directory.")

    # --- Tests for ChromaDB Error Handling and New Reflection Loading ---

    @patch('utils.chroma_utils.get_chroma_client', return_value=None)
    def test_08_load_reflections_chroma_unavailable(self, mock_get_client):
        """Test handle_load_reflections when ChromaDB client fails to initialize."""
        print(f"\nRunning test: test_08_load_reflections_chroma_unavailable in {os.getcwd()}")

        async def run_async_test():
            # Need to initialize the project first so StateManager can try to load status
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            init_result = await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            self.assertEqual(init_result.get("status"), "success", msg="Initialization failed")

            # Attempt to load reflections
            print("Attempting to load reflections with mocked unavailable Chroma client...")
            load_result = await handle_load_reflections(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print(f"Load reflections result: {load_result}")

            # Assertions
            mock_get_client.assert_called_once() # Check if the mock was used
            self.assertEqual(load_result.get("status"), "error")
            self.assertIn("ChromaDB client is not available", load_result.get("message", ""))
            self.assertEqual(load_result.get("summary", "not_empty"), "") # Summary should be empty on error

        asyncio.run(run_async_test())

    @patch('utils.state_manager.StateManager') # Patch StateManager used within the handler
    def test_09_load_reflections_success_empty(self, MockStateManager):
        """Test handle_load_reflections success path when ChromaDB returns no reflections."""
        print(f"\nRunning test: test_09_load_reflections_success_empty in {os.getcwd()}")

        # Configure the mock StateManager instance that will be created inside the handler
        mock_state_manager_instance = MockStateManager.return_value
        # Make the get_all_reflections method return an empty list
        mock_state_manager_instance.get_all_reflections = AsyncMock(return_value=[])

        async def run_async_test():
            # Initialize project (needed for directory structure, though StateManager is mocked)
            print(f"Initializing project in: {self.TEST_DIR.resolve()} (structure only)")
            # We don't need the real init side effects here as StateManager is mocked
            if not self.TEST_DIR.exists(): self.TEST_DIR.mkdir()
            if not (self.TEST_DIR / ".chungoid").exists(): (self.TEST_DIR / ".chungoid").mkdir()
            print("Mock project initialized.")

            # Call load_reflections
            print("Calling load_reflections with mocked StateManager (empty result)...")
            # Need to make handle_load_reflections use the patched StateManager
            # One way is to patch StateManager globally or use dependency injection if available
            # Assuming the handler instantiates StateManager internally for this test setup:
            load_result = await handle_load_reflections(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print(f"Load reflections result: {load_result}")

            # Check that StateManager was instantiated correctly within the handler call scope
            MockStateManager.assert_called_once()
            # Check that get_all_reflections was called on the instance
            mock_state_manager_instance.get_all_reflections.assert_called_once()

            # Expect success response with empty data
            self.assertEqual(load_result.status_code, 200, "Expected HTTP 200 status code")
            body = json.loads(load_result.body.decode())
            self.assertEqual(body.get("status"), "success", "Expected success status in response")
            self.assertEqual(body.get("reflections"), [], "Expected empty reflections list")
            print("Load reflections succeeded with empty data.")

        asyncio.run(run_async_test())

    @patch('utils.state_manager.StateManager') # Patch StateManager used within the handler
    def test_10_load_reflections_success_with_data(self, MockStateManager):
        """Test handle_load_reflections success path when ChromaDB returns reflections."""
        print(f"\nRunning test: test_10_load_reflections_success_with_data in {os.getcwd()}")

        # Sample data that get_all_reflections would return
        sample_reflections = [
            {
                'id': 'uuid1',
                'metadata': {'stage_number': 1.0, 'timestamp': '2025-01-01T10:00:00Z'},
                'document': 'Reflection from stage 1',
                'timestamp': '2025-01-01T10:00:00Z' # Added by get_all_reflections
            },
            {
                'id': 'uuid2',
                'metadata': {'stage_number': 0.0, 'timestamp': '2025-01-01T09:00:00Z'},
                'document': 'Reflection from stage 0',
                'timestamp': '2025-01-01T09:00:00Z' # Added by get_all_reflections
            }
        ]

        # Configure the mock StateManager instance
        mock_state_manager_instance = MockStateManager.return_value
        mock_state_manager_instance.get_all_reflections = AsyncMock(return_value=sample_reflections)

        async def run_async_test():
            # Initialize project
            print(f"Initializing project in: {self.TEST_DIR.resolve()} (structure only)")
            if not self.TEST_DIR.exists(): self.TEST_DIR.mkdir()
            if not (self.TEST_DIR / ".chungoid").exists(): (self.TEST_DIR / ".chungoid").mkdir()
            print("Mock project initialized.")

            # Call load_reflections
            print("Calling load_reflections with mocked StateManager (with data)...")
            load_result = await handle_load_reflections(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            print(f"Load reflections result: {load_result}")

            MockStateManager.assert_called()
            mock_state_manager_instance.get_all_reflections.assert_called_once_with(limit=20)
            self.assertEqual(load_result.status_code, 200, "Expected HTTP 200 status code")
            body = json.loads(load_result.body.decode())
            self.assertEqual(body.get("status"), "success", "Expected success status in response")
            summary = load_result.get("summary", "")
            self.assertIn(f"Summary of last {len(sample_reflections)} reflections", summary)
            self.assertIn("(limit: 20)", summary)
            self.assertIn("Stage 1.0", summary)
            self.assertIn("Stage 0.0", summary)
            self.assertIn("2025-01-01T10:00:00Z", summary)
            self.assertIn("2025-01-01T09:00:00Z", summary)

        asyncio.run(run_async_test())

    # --- Tests for handle_retrieve_reflections_tool --- #

    @patch('utils.chroma_utils.get_chroma_client', return_value=None)
    def test_11_retrieve_reflections_chroma_unavailable(self, mock_get_client):
        """Test retrieve_reflections handles ChromaDB unavailable."""
        print(f"\nRunning test: test_11_retrieve_reflections_chroma_unavailable in {os.getcwd()}")
        async def run_async_test():
            # Initialize project
            init_result = await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            self.assertEqual(init_result.get("status"), "success", msg="Initialization failed")

            # Attempt to retrieve reflections
            print("Attempting to retrieve reflections with mocked unavailable Chroma client...")
            retrieve_result = await handle_retrieve_reflections_tool(
                target_directory=str(self.TEST_DIR.resolve()),
                query="test query",
                ctx=None
            )
            print(f"Retrieve reflections result: {retrieve_result}")

            # Expect an error response (check dict content, not status_code)
            # <<< MODIFIED ASSERTIONS >>>
            self.assertIsInstance(retrieve_result, dict, "Expected dict response on error")
            self.assertEqual(retrieve_result.get("status"), "error", "Expected error status in response dict")
            self.assertIn("ChromaDB operation failed", retrieve_result.get("error", ""), "Expected ChromaDB error message")
            print("Retrieve reflections correctly handled ChromaDB unavailability.")

        asyncio.run(run_async_test())

    @patch('utils.state_manager.StateManager')
    def test_12_retrieve_reflections_success_empty(self, MockStateManager):
        """Test handle_retrieve_reflections_tool success path with no results."""
        print(f"\nRunning test: test_12_retrieve_reflections_success_empty in {os.getcwd()}")
        mock_sm_instance = MockStateManager.return_value
        mock_sm_instance.get_reflection_context_from_chroma.return_value = [] # Empty list

        async def run_async_test():
            # Initialize project (structure only)
            print(f"Initializing project in: {self.TEST_DIR.resolve()} (structure only)")
            if not self.TEST_DIR.exists(): self.TEST_DIR.mkdir()
            if not (self.TEST_DIR / ".chungoid").exists(): (self.TEST_DIR / ".chungoid").mkdir()
            print("Mock project initialized.")

            # Call retrieve_reflections
            print("Calling retrieve_reflections with mocked StateManager (empty result)...")
            retrieve_result = await handle_retrieve_reflections_tool(
                target_directory=str(self.TEST_DIR.resolve()),
                query="test query",
                n_results=3,
                ctx=None
            )
            print(f"Retrieve reflections result: {retrieve_result}")

            MockStateManager.assert_called_once()
            mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(
                query="test query", n_results=3, filter_stage_min=None
            )

            # Expect success response with empty list
            self.assertEqual(retrieve_result.status_code, 200, "Expected HTTP 200 status code")
            body = json.loads(retrieve_result.body.decode())
            self.assertEqual(body.get("status"), "success", "Expected success status in response")
            self.assertEqual(body.get("results"), [], "Expected empty results list")
            print("Retrieve reflections succeeded with empty data.")

        asyncio.run(run_async_test())

    @patch('utils.state_manager.StateManager')
    def test_13_retrieve_reflections_success_with_data(self, MockStateManager):
        """Test handle_retrieve_reflections_tool success path with sample data."""
        print(f"\nRunning test: test_13_retrieve_reflections_success_with_data in {os.getcwd()}")
        mock_sm_instance = MockStateManager.return_value
        mock_retrieved_data = [
            {"id": "r1", "metadata": {"stage": 0.0}, "document": "Retrieved 1"}
        ]
        mock_sm_instance.get_reflection_context_from_chroma.return_value = mock_retrieved_data

        async def run_async_test():
            # Initialize project (structure only)
            print(f"Initializing project in: {self.TEST_DIR.resolve()} (structure only)")
            if not self.TEST_DIR.exists(): self.TEST_DIR.mkdir()
            if not (self.TEST_DIR / ".chungoid").exists(): (self.TEST_DIR / ".chungoid").mkdir()
            print("Mock project initialized.")

            # Call retrieve_reflections
            print("Calling retrieve_reflections with mocked StateManager (with data)...")
            retrieve_result = await handle_retrieve_reflections_tool(
                target_directory=str(self.TEST_DIR.resolve()),
                query="another query",
                n_results=1,
                filter_stage_min="0.0",
                ctx=None
            )
            print(f"Retrieve reflections result: {retrieve_result}")

            MockStateManager.assert_called_once()
            mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(
                query="another query", n_results=1, filter_stage_min=0.0 # Should convert str to float
            )

            # Expect success response with data
            self.assertEqual(retrieve_result.status_code, 200, "Expected HTTP 200 status code")
            body = json.loads(retrieve_result.body.decode())
            self.assertEqual(body.get("status"), "success", "Expected success status in response")
            self.assertEqual(body.get("results"), mock_retrieved_data, "Expected retrieved data")
            print("Retrieve reflections succeeded with data.")

        asyncio.run(run_async_test())

    # --- Tests for handle_submit_stage_artifacts --- #

    @patch('utils.state_manager.StateManager')
    def test_14_submit_artifacts_chroma_error_on_store(self, MockStateManager):
        """Test handle_submit_stage_artifacts when storing artifact context fails."""
        print(f"\nRunning test: test_14_submit_artifacts_chroma_error_on_store in {os.getcwd()}")
        mock_sm_instance = MockStateManager.return_value
        # Simulate _store_reflections_in_chroma raising an error
        mock_sm_instance._store_reflections_in_chroma.side_effect = ChromaOperationError("Test Chroma Store Error")
        # Make update_status succeed so we reach the chroma part
        mock_sm_instance.update_status.return_value = True

        async def run_async_test():
            # Initialize project (structure only)
            print(f"Initializing project in: {self.TEST_DIR.resolve()} (structure only)")
            if not self.TEST_DIR.exists(): self.TEST_DIR.mkdir()
            if not (self.TEST_DIR / ".chungoid").exists(): (self.TEST_DIR / ".chungoid").mkdir()
            print("Mock project initialized.")

            # Call submit_artifacts
            print("Calling submit_artifacts with mock causing Chroma store error...")
            artifacts = {"dev-docs/reflections.md": "Some reflections"}
            submit_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=1,
                generated_artifacts=artifacts,
                stage_result_status="PASS",
                ctx=None
            )
            print(f"Submit artifacts result: {submit_result}")

            MockStateManager.assert_called_once()
            mock_sm_instance.update_status.assert_called_once()
            # Check _store_reflections_in_chroma was called before it raised the error
            mock_sm_instance._store_reflections_in_chroma.assert_called_once()

            # Expect an error response because the decorator catches ChromaOperationError
            self.assertEqual(submit_result.status_code, 500, "Expected HTTP 500 status code")
            body = json.loads(submit_result.body.decode())
            self.assertEqual(body.get("status"), "error", "Expected error status in response")
            self.assertIn("ChromaDB operation failed", body.get("error", ""), "Expected ChromaDB error message")
            print("Submit artifacts handled Chroma store error correctly.")

        asyncio.run(run_async_test())

    # --- Tests for handle_get_file --- #

    def test_15_get_file(self):
        """Test the get_file tool handler for success and failure cases."""
        print(f"\nRunning test: test_15_get_file in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize Project and Create Test File ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            init_result = await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()), ctx=None)
            self.assertEqual(init_result.get("status"), "success", msg="Initialization failed")
            print("Project initialized.")

            # Create a dummy file
            docs_dir = self.TEST_DIR / "docs"
            docs_dir.mkdir(exist_ok=True)
            test_file_path = docs_dir / "myfile.txt"
            test_content = "This is the content of the test file.\nLine 2."
            with open(test_file_path, "w") as f:
                f.write(test_content)
            print(f"Created test file: {test_file_path}")

            # --- Test Success Case ---
            print("Testing get_file success case...")
            success_result = await handle_get_file(
                target_directory=str(self.TEST_DIR.resolve()),
                relative_path="docs/myfile.txt",
                ctx=None,
            )
            print(f"Success result: {success_result}")
            self.assertEqual(success_result.get("status"), "success", msg="Get file should succeed")
            self.assertEqual(
                success_result.get("content"),
                test_content,
                msg="File content does not match",
            )
            print("Get file success case passed.")

            # --- Test File Not Found Case ---
            print("Testing get_file not found case...")
            not_found_result = await handle_get_file(
                target_directory=str(self.TEST_DIR.resolve()),
                relative_path="docs/nosuchfile.txt",
                ctx=None,
            )
            print(f"Not found result: {not_found_result}")
            self.assertEqual(not_found_result.get("status"), "error", msg="Get file should fail")
            self.assertIn(
                "File not found",
                not_found_result.get("error", ""),
                msg="Expected 'File not found' error",
            )
            print("Get file not found case passed.")

            # --- Test Path Traversal Attempt ---
            print("Testing get_file path traversal case...")
            # Need handle_get_file definition to call it
            from chungoidmcp import handle_get_file # Ensure imported

            traversal_result = await handle_get_file(
                target_directory=str(self.TEST_DIR.resolve()),
                relative_path="../outside.txt", # Attempt to go outside project dir
                ctx=None,
            )
            print(f"Traversal result: {traversal_result}")
            self.assertEqual(traversal_result.get("status"), "error", msg="Path traversal should fail")
            self.assertIn(
                "Invalid path", # Check for generic invalid path error
                traversal_result.get("error", ""),
                msg="Expected invalid path error message",
            )
            print("Get file path traversal case passed.")

        asyncio.run(run_async_test())


# This allows running the tests from the command line
if __name__ == "__main__":
    # Configure logging for tests (optional, might be useful)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
