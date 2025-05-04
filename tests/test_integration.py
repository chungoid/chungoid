import unittest
import os
import shutil
from pathlib import Path
import sys
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
import chromadb
import logging

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
    # We'll need a way to simulate the context or pass necessary args
)
from utils.state_manager import StateManager  # Assuming state_manager might be needed directly


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
        """Test adding and retrieving data via StateManager's ChromaDB interactions."""
        print(f"\nRunning test: test_07_chromadb_operations in {os.getcwd()}")

        async def run_async_test():
            # <<< Apply patch manually within the test using a context manager >>>
            with patch(
                "utils.chroma_utils.get_chroma_client", new_callable=AsyncMock
            ) as mock_get_chroma_client:
                # --- Setup Mock Chroma Client (inside context manager) --- #
                mock_client_instance = AsyncMock(spec=chromadb.ClientAPI)  # Client getter IS async
                mock_get_chroma_client.return_value = mock_client_instance
                # Collection methods are SYNCHRONOUS, use MagicMock
                mock_collection_artifacts = MagicMock(spec=chromadb.Collection)
                mock_collection_reflections = MagicMock(spec=chromadb.Collection)

                # Mock get_or_create_collection AND get_collection based on name
                def collection_factory(name, **kwargs):
                    self.logger.info(f"Mock collection_factory called for name: {name}")  # Add log
                    if name == state_manager._CONTEXT_COLLECTION_NAME:
                        return mock_collection_artifacts
                    elif name == state_manager._REFLECTIONS_COLLECTION_NAME:
                        return mock_collection_reflections
                    else:
                        # Return a default mock if name doesn't match expected,
                        # or raise error depending on desired test strictness.
                        # Raising error is stricter:
                        raise ValueError(
                            f"Mock collection_factory received unexpected collection name: {name}"
                        )

                mock_client_instance.get_or_create_collection.side_effect = collection_factory
                mock_client_instance.get_collection.side_effect = (
                    collection_factory  # <<< Mock get_collection too
                )

                # Mock add/get/query methods on the *collection* mocks
                mock_collection_artifacts.add = MagicMock(return_value=None)
                mock_collection_artifacts.get = MagicMock(
                    return_value={  # Mock return value for get
                        "ids": [
                            "stage_1.0_dev-docs/design/interface.yaml",
                            "stage_1.0_src/module.py",
                        ],
                        "documents": ["content: interface details", "# Python code"],
                        "metadatas": [
                            {"stage": 1.0, "filename": "dev-docs/design/interface.yaml"},
                            {"stage": 1.0, "filename": "src/module.py"},
                        ],
                    }
                )
                mock_collection_reflections.add = MagicMock(return_value=None)
                mock_collection_reflections.query = MagicMock(
                    return_value={  # Mock return value for query
                        "ids": [["some_id"]],
                        "documents": [["Stage 1 reflection content."]],
                        "metadatas": [[{"stage": 1.0, "status": "PASS"}]],
                        "distances": [[0.1]],
                    }
                )

                # --- Setup: Initialize Project (needed for StateManager init) ---
                await handle_initialize_project(
                    target_directory=str(self.TEST_DIR.resolve()), ctx=None
                )
                print("Project initialized for ChromaDB test.")

                # --- Initialize StateManager pointing to test directory --- #
                server_stages_dir = project_root / "server_prompts" / "stages"
                state_manager = StateManager(
                    target_directory=str(self.TEST_DIR.resolve()),
                    server_stages_dir=str(server_stages_dir),
                )

                # --- Test Persist Artifacts --- #
                print("Testing store_artifact_context_in_chroma...")
                stage_num = 1.0
                artifacts_to_persist = {
                    "dev-docs/design/interface.yaml": "content: interface details",
                    "src/module.py": "# Python code",
                }
                # Call the correct method store_artifact_context_in_chroma individually
                for rel_path, content in artifacts_to_persist.items():
                    await state_manager.store_artifact_context_in_chroma(
                        stage_number=stage_num,
                        rel_path=rel_path,
                        content=content,
                        # artifact_type can be inferred or passed explicitly if needed by test
                    )

                mock_get_chroma_client.assert_called_once()
                # Assert using the correct collection name
                mock_client_instance.get_or_create_collection.assert_called_with(
                    name=state_manager._CONTEXT_COLLECTION_NAME
                )
                self.assertTrue(mock_collection_artifacts.add.called)
                self.assertEqual(mock_collection_artifacts.add.call_count, 2)
                call_args, call_kwargs = mock_collection_artifacts.add.call_args
                self.assertEqual(len(call_kwargs.get("ids", [])), 2)
                self.assertEqual(len(call_kwargs.get("documents", [])), 2)
                self.assertEqual(len(call_kwargs.get("metadatas", [])), 2)
                print("store_artifact_context_in_chroma assertions passed.")

                # --- Test Get Artifacts by ID --- #
                mock_collection_artifacts.add.reset_mock()
                mock_collection_artifacts.get.reset_mock()
                mock_client_instance.get_or_create_collection.reset_mock()
                mock_client_instance.get_collection.reset_mock()  # <<< Reset get_collection mock too
                # DO NOT reset mock_get_chroma_client yet - it should be cached now

                print("Testing get_artifact_context_from_chroma_by_ids...")
                doc_ids_to_get = [
                    "stage_1.0_dev-docs/design/interface.yaml",
                    "stage_1.0_src/module.py",
                ]
                # Don't need to mock get return value here anymore, done above

                retrieved_context = await state_manager.get_artifact_context_from_chroma_by_ids(
                    doc_ids_to_get
                )

                # Assertions
                mock_get_chroma_client.assert_called_once()  # Check it *still* has only been called once
                # Assert using the correct collection name (StateManager uses get_collection here)
                mock_client_instance.get_collection.assert_called_once_with(
                    name=state_manager._CONTEXT_COLLECTION_NAME
                )
                # Now this assertion should pass because get_collection returned the correct mock collection
                mock_collection_artifacts.get.assert_called_once_with(
                    ids=doc_ids_to_get, include=["documents", "metadatas"]
                )
                # Fix assertion: check if it's a list, not a string
                self.assertIsInstance(retrieved_context, list)
                # Check the contents of the list
                self.assertEqual(len(retrieved_context), 2)
                self.assertEqual(retrieved_context[0]["id"], doc_ids_to_get[0])
                self.assertEqual(retrieved_context[0]["document"], "content: interface details")
                self.assertEqual(retrieved_context[1]["id"], doc_ids_to_get[1])
                self.assertEqual(retrieved_context[1]["document"], "# Python code")
                # Removed the string containment checks as they are less specific
                print("get_artifact_context_from_chroma_by_ids assertions passed.")

                # --- Test Add/Get Reflection --- #
                print("Testing add_reflection and get_reflections...")
                # Reset mocks specifically for reflection tests
                mock_collection_reflections.add.reset_mock()
                mock_collection_reflections.query.reset_mock()
                mock_client_instance.get_or_create_collection.reset_mock()
                mock_client_instance.get_collection.reset_mock()  # <<< Reset get_collection mock here too
                # get_chroma_client should still be cached

                reflection_text = "Stage 1 reflection content."
                reflection_stage = 1.0
                reflection_status = "PASS"
                # Don't need to mock add return value here anymore, done above

                reflection_id = await state_manager.add_reflection(
                    reflection_stage, reflection_status, reflection_text
                )

                mock_get_chroma_client.assert_called_once()  # Still only called once overall
                # Assert using the correct collection name (StateManager uses get_or_create_collection here)
                mock_client_instance.get_or_create_collection.assert_called_once_with(
                    name=state_manager._REFLECTIONS_COLLECTION_NAME
                )
                self.assertTrue(mock_collection_reflections.add.called)
                call_args_refl, call_kwargs_refl = mock_collection_reflections.add.call_args
                self.assertIsNotNone(call_kwargs_refl.get("ids"))
                self.assertIn(reflection_text, call_kwargs_refl.get("documents", ["missing"])[0])
                self.assertEqual(
                    call_kwargs_refl.get("metadatas", [{}])[0].get("stage"), reflection_stage
                )
                print("add_reflection assertions passed.")

                # Test get_reflections
                query_text = "reflection content"
                # Don't need to mock query return value here anymore, done above

                retrieved_reflections = await state_manager.get_reflections(
                    query=query_text, n_results=1
                )

                # Assert using the correct collection name (StateManager uses get_collection here)
                mock_client_instance.get_collection.assert_called_with(
                    name=state_manager._REFLECTIONS_COLLECTION_NAME
                )
                mock_collection_reflections.query.assert_called_once()
                # Check some query args (be careful with embedding mocks if applicable)
                call_args_q, call_kwargs_q = mock_collection_reflections.query.call_args
                self.assertEqual(call_kwargs_q.get("query_texts"), [query_text])
                self.assertEqual(call_kwargs_q.get("n_results"), 1)

                self.assertIsInstance(retrieved_reflections, list)
                self.assertEqual(len(retrieved_reflections), 1)
                self.assertEqual(retrieved_reflections[0].get("document"), reflection_text)
                self.assertEqual(
                    retrieved_reflections[0].get("metadata", {}).get("stage"), reflection_stage
                )
                print("get_reflections assertions passed.")

        # Run the async test function
        asyncio.run(run_async_test())


if __name__ == "__main__":
    unittest.main()
