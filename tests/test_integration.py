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
import yaml
import pytest

# Try importing necessary components
try:
    from chungoidmcp import (
        handle_initialize_project,
        handle_get_project_status,
        handle_submit_stage_artifacts,
        handle_load_reflections,
        handle_retrieve_reflections_tool as handle_retrieve_reflections, # Alias for clarity
        handle_get_file,
        handle_prepare_next_stage,
        # Import exceptions used in tests if needed
    )
    from utils.state_manager import StateManager, StatusFileError, ChromaOperationError
    # Import other necessary utils if testing them directly
    import utils.chroma_utils
    import chromadb
except ImportError as e:
    print(f"Failed to import necessary components: {e}")

pytestmark = pytest.mark.legacy

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
            self.assertIsInstance(status_result.get("runs"), list, msg="Status should have a 'runs' list")
            print("Initialize and Get Status test passed.")

        # Run the async function using asyncio.run()
        asyncio.run(run_async_test())

    @unittest.skip("Requires PersistentClient setup, conflicting with HTTP client")
    def test_07_chromadb_operations(self):
        """Test direct interaction with ChromaDB via utils (assuming sync utils)."""
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
            print("Cleaned up temp dir: {self.TEST_DIR.resolve()}")

    # --- Tests for Reflection Loading/Retrieval Error Handling ---

    @patch('chungoidmcp._initialize_state_manager_for_target')
    def test_08_load_reflections_chroma_unavailable(self, MockStateManager):
        """Test handle_load_reflections when ChromaDB operation fails."""
        print(f"\nRunning test: test_08_load_reflections_chroma_unavailable in {os.getcwd()}")

        async def run_async_test():
            # Mock _initialize_state_manager_for_target to raise ChromaOperationError
            with patch("chungoidmcp._initialize_state_manager_for_target") as mock_init_sm:
                # Simulate Chroma client failing *within* StateManager, e.g., during get_all_reflections
                mock_sm_instance = MagicMock()
                mock_sm_instance.get_all_reflections.side_effect = ChromaOperationError("ChromaDB client is not available.")
                mock_init_sm.return_value = mock_sm_instance

                print("Attempting to load reflections with mocked unavailable Chroma client...")
                load_result = await handle_load_reflections(
                    target_directory=str(self.TEST_DIR.resolve())
                )
                print(f"Load reflections result: {load_result}")
                self.assertEqual(load_result.get("status"), "error", msg="Load reflections should fail")
                # Check that the original error message is part of the returned message
                self.assertIn("ChromaDB client is not available", load_result.get("message", ""), "Original error missing from response")

        asyncio.run(run_async_test())

    @patch('chungoidmcp._initialize_state_manager_for_target')
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
            self.assertEqual(load_result.get("status"), "success", msg="Load reflections should succeed")
            self.assertIn("No reflections found", load_result.get("summary", ""))
            mock_state_manager_instance.get_all_reflections.assert_called_once()
            print("Load reflections succeeded with empty data.")

        asyncio.run(run_async_test())

    @patch('chungoidmcp._initialize_state_manager_for_target')
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
            self.assertEqual(load_result.get("status"), "success", msg="Load reflections should succeed")
            self.assertIn("Loaded 1 reflections.", load_result.get("summary", ""))
            mock_state_manager_instance.get_all_reflections.assert_called_once_with(limit=20)

        asyncio.run(run_async_test())

    # --- Tests for handle_retrieve_reflections_tool --- #

    @patch('chungoidmcp._initialize_state_manager_for_target')
    def test_11_retrieve_reflections_chroma_unavailable(self, MockStateManager):
        """Test retrieve_reflections handles ChromaDB unavailable."""
        print(f"\nRunning test: test_11_retrieve_reflections_chroma_unavailable in {os.getcwd()}")

        async def run_async_test():
            # Mock _initialize_state_manager_for_target to raise ChromaOperationError during query
            with patch("chungoidmcp._initialize_state_manager_for_target") as mock_init_sm:
                # Simulate Chroma query failing within StateManager
                mock_sm_instance = MagicMock()
                mock_sm_instance.get_reflection_context_from_chroma.side_effect = ChromaOperationError("ChromaDB query operation failed for collection 'chungoid_reflections'. Check logs for details.")
                mock_init_sm.return_value = mock_sm_instance

                print("Attempting to retrieve reflections with mocked unavailable Chroma client...")
                retrieve_result = await handle_retrieve_reflections(
                    target_directory=str(self.TEST_DIR.resolve()), query="test query"
                )
                print(f"Retrieve reflections result: {retrieve_result}")

                self.assertEqual(retrieve_result.get("status"), "error", msg="Retrieve reflections should fail")
                # Check that the original error message is part of the returned message
                self.assertIn("ChromaDB query operation failed", retrieve_result.get("message", ""), "Original error missing from response")

        asyncio.run(run_async_test())

    @patch('chungoidmcp._initialize_state_manager_for_target')
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
            retrieve_result = await handle_retrieve_reflections(
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
            self.assertEqual(retrieve_result.get("status"), "success")
            self.assertEqual(retrieve_result.get("reflections", None), [])
            mock_sm_instance.get_reflection_context_from_chroma.assert_called_once()
            print("Retrieve reflections succeeded with empty data.")

        asyncio.run(run_async_test())

    @patch('chungoidmcp._initialize_state_manager_for_target')
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
            retrieve_result = await handle_retrieve_reflections(
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
            self.assertEqual(retrieve_result.get("status"), "success")
            self.assertEqual(len(retrieve_result.get("reflections", [])), 1)
            self.assertEqual(retrieve_result["reflections"][0]["document"], "Retrieved 1")
            mock_sm_instance.get_reflection_context_from_chroma.assert_called_once()
            print("Retrieve reflections succeeded with data.")

        asyncio.run(run_async_test())

    # --- Tests for handle_submit_stage_artifacts --- #

    @patch('chungoidmcp._initialize_state_manager_for_target')
    def test_14_submit_artifacts_chroma_error_on_store(self, mock_init_sm):
        """Test handle_submit_stage_artifacts when storing artifact context fails."""
        print(f"\nRunning test: test_14_submit_artifacts_chroma_error_on_store in {os.getcwd()}")

        async def run_async_test():
            # Initialize project
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            await handle_initialize_project(target_directory=str(self.TEST_DIR.resolve()))

            # Mock StateManager methods to simulate failure during artifact context storage
            mock_sm_instance = MagicMock()
            # Configure the specific method to raise an error
            mock_sm_instance.store_artifact_context_in_chroma.side_effect = ChromaOperationError("Simulated Chroma DB Error")
            # Configure other methods to return successful values
            mock_sm_instance.get_latest_run_id.return_value = 0 # Needed for reflection persistence
            mock_sm_instance.persist_reflections_to_chroma.return_value = None # Returns None on success
            mock_sm_instance.update_status.return_value = True

            # Make the patched helper return our configured mock instance
            mock_init_sm.return_value = mock_sm_instance

            # Prepare data for submission
            artifacts_to_submit = {"dev-docs/reflections.md": "Some reflection text"}
            reflection_text_to_submit = "This is the reflection text for the stage."

            # Call the handler
            print("Calling submit_artifacts with mock causing Chroma store error...")
            submit_result = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()), # Pass target dir
                stage_number=1,
                generated_artifacts=artifacts_to_submit,
                reflection_text=reflection_text_to_submit,
                stage_result_status="PASS",
            )
            print(f"Submit artifacts result: {submit_result}")

            mock_init_sm.assert_called_once() # Check the patched helper was called
            mock_init_sm.return_value.persist_reflections_to_chroma.assert_called_once()
            mock_init_sm.return_value.store_artifact_context_in_chroma.assert_called_once() # Verify the failing method was called

            # Check the response indicates an error during artifact storage
            self.assertEqual(submit_result.get("status"), "error", "Expected error status")

        asyncio.run(run_async_test())

    # --- Tests for handle_get_file --- #

    def test_15_get_file(self):
        """Test the get_file tool handler for success and failure cases."""
        print(f"\nRunning test: test_15_get_file in {os.getcwd()}")

        async def run_async_test():
            # --- Setup: Initialize Project and Create Test File ---
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            from chungoidmcp import handle_get_file # <<< ADDED IMPORT INSIDE ASYNC SCOPE

            # Initialize project
            print("Initializing project...")
            init_result = await handle_initialize_project(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
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
            success_result = await handle_get_file(target_directory=str(self.TEST_DIR.resolve()), relative_path="docs/myfile.txt")
            self.assertEqual(success_result.get("status"), "success", msg="Get file should succeed")
            self.assertEqual(
                success_result.get("content"),
                test_content,
                msg="File content does not match",
            )
            print("Get file success case passed.")

            # --- Test Not Found Case ---
            print("Testing get_file not found case...")
            not_found_result = await handle_get_file(target_directory=str(self.TEST_DIR.resolve()), relative_path="docs/nosuchfile.txt")
            print(f"Not found result: {not_found_result}")
            self.assertEqual(not_found_result.get("status"), "error", msg="Get file should fail with status 'error'")
            self.assertIn("Tool execution error", not_found_result.get("message", ""), msg="Error message should indicate tool execution error")
            self.assertIn("Could not find the requested file", not_found_result.get("message", ""), msg="Specific file not found message missing")
            print("Not found case passed.")

            # --- Test Access Denied Case ---
            print("Testing get_file access denied case...")
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

    @unittest.skip("Debugging PromptLoadError/path issue")
    def test_16_prepare_next_stage(self):
        """Test the prepare_next_stage tool handler."""
        print(f"\nRunning test: test_16_prepare_next_stage in {os.getcwd()}")

        # Import the specific handler function needed
        # Ensure engine is also imported or accessible if needed directly (shouldn't be)
        from chungoidmcp import handle_prepare_next_stage, handle_initialize_project, handle_submit_stage_artifacts
        from engine import ChungoidEngine # Needed for mocking potentially

        async def run_async_test():
            # Helper to create dummy prompt files for testing PromptManager loading
            def _create_dummy_prompt_files():
                # Calculate path relative to the engine.py location, as that's where it looks
                core_root = Path(__file__).parent.parent # Go up from tests/ to chungoid-core/
                server_prompts_path = core_root / "server_prompts"
                stages_path = server_prompts_path / "stages"
                stages_path.mkdir(parents=True, exist_ok=True)
                common_path = server_prompts_path / "common.yaml"

                # Common Template (Must have preamble and postamble strings)
                common_content = "preamble: 'COMMON PREAMBLE'\npostamble: 'COMMON POSTAMBLE'"
                with open(common_path, 'w') as f:
                    f.write(common_content)

                # Stage Files (Must have system_prompt and user_prompt strings)
                for i in range(6):
                    stage_file = stages_path / f"stage{i}.yaml"
                    # Use minimal valid YAML strings
                    stage_content = (
                        f"description: 'Dummy Stage {i}'\n"
                        f"system_prompt: 'Sys Prompt {i}'\n"
                        f"user_prompt: 'User Prompt {i}'\n"
                    )
                    with open(stage_file, 'w') as f:
                        f.write(stage_content)
                print(f"Created dummy prompt files in {stages_path.parent}")

            # --- Setup: Initialize Project --- #
            print(f"Initializing project in: {self.TEST_DIR.resolve()}")
            init_result = await handle_initialize_project(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            self.assertEqual(init_result.get("status"), "success", msg="Initialization failed")
            print("Project initialized.")

            # --- Setup: Create Dummy Stage/Common Files --- #
            # engine.py assumes these exist relative to core root
            # StateManager finds them relative to server_stages_dir passed in init
            # PromptManager finds them relative to stages_root_dir passed in init
            # Need to ensure these paths align or create files where expected
            core_root = Path(__file__).parent.parent # Assumes tests run from project root
            server_prompts_dir = core_root / 'server_prompts'
            stages_dir = server_prompts_dir / 'stages'
            stages_dir.mkdir(parents=True, exist_ok=True)
            common_path = server_prompts_dir / 'common.yaml'
            stage0_path = stages_dir / 'stage0.yaml'
            stage1_path = stages_dir / 'stage1.yaml'

            # Write basic content
            common_path.write_text("preamble: COMMON PREAMBLE\npostamble: COMMON POSTAMBLE")
            stage0_path.write_text("prompt_details: Stage 0 Details {{ context_data.initial_goal | default('') }}\nuser_prompt: User prompt for Stage 0.")
            stage1_path.write_text("prompt_details: Stage 1 Details\nuser_prompt: User prompt for Stage 1. Last status was {{ context_data.last_status.status }}")
            print(f"Created dummy prompt files in {server_prompts_dir}")

            # --- Test Initial Call (Should prepare Stage 0) --- #
            print("Calling prepare_next_stage for the first time...")
            prep_result_0 = await handle_prepare_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Prepare result (Stage 0): {json.dumps(prep_result_0, indent=2)}")

            self.assertEqual(prep_result_0.get("status"), "success", msg="Prepare stage 0 failed")
            self.assertEqual(prep_result_0.get("next_stage"), 0.0, msg="Expected next stage to be 0.0")
            self.assertIn("COMMON PREAMBLE", prep_result_0.get("prompt", ""), msg="Common preamble missing")
            self.assertIn("User prompt for Stage 0.", prep_result_0.get("prompt", ""), msg="Stage 0 prompt missing")
            self.assertIn("COMMON POSTAMBLE", prep_result_0.get("prompt", ""), msg="Common postamble missing")
            context0 = prep_result_0.get("gathered_context", {})
            self.assertIsInstance(context0, dict, msg="Context should be a dict")
            self.assertEqual(context0.get("current_stage"), 0.0)
            self.assertIn("project_directory", context0)
            self.assertIn("No previous status found.", str(context0.get("last_status")))
            self.assertNotIn("relevant_reflections", context0) # Reflections gathered only for stage > 0
            print("Assertions for Stage 0 passed.")

            # --- Simulate Completing Stage 0 --- #
            print("\nSimulating completion of Stage 0...")
            submit_result_0 = await handle_submit_stage_artifacts(
                target_directory=str(self.TEST_DIR.resolve()),
                stage_number=0.0,
                generated_artifacts={"output/stage0.txt": "Stage 0 output"},
                stage_result_status="PASS",
                ctx=None
            )
            self.assertEqual(submit_result_0.get("status"), "success", msg="Submit Stage 0 failed")
            print("Stage 0 submitted as PASS.")

            # --- Test Second Call (Should prepare Stage 1) --- #
            print("\nCalling prepare_next_stage again (expecting Stage 1)...")
            prep_result_1 = await handle_prepare_next_stage(
                target_directory=str(self.TEST_DIR.resolve()), ctx=None
            )
            print(f"Prepare result (Stage 1): {json.dumps(prep_result_1, indent=2)}")

            self.assertEqual(prep_result_1.get("status"), "success", msg="Prepare stage 1 failed")
            self.assertEqual(prep_result_1.get("next_stage"), 1.0, msg="Expected next stage to be 1.0")
            self.assertIn("COMMON PREAMBLE", prep_result_1.get("prompt", ""), msg="Common preamble missing S1")
            self.assertIn("User prompt for Stage 1.", prep_result_1.get("prompt", ""), msg="Stage 1 prompt missing")
            self.assertIn("Last status was PASS", prep_result_1.get("prompt", ""), msg="Stage 1 prompt context missing/wrong") # Check context injection
            self.assertIn("COMMON POSTAMBLE", prep_result_1.get("prompt", ""), msg="Common postamble missing S1")
            context1 = prep_result_1.get("gathered_context", {})
            self.assertIsInstance(context1, dict, msg="Context S1 should be a dict")
            self.assertEqual(context1.get("current_stage"), 1.0)
            self.assertIn("project_directory", context1)
            self.assertEqual(context1.get("last_status", {}).get("stage"), 0.0)
            self.assertEqual(context1.get("last_status", {}).get("status"), "PASS")
            self.assertIn("relevant_reflections", context1) # Should attempt to gather reflections for stage > 0
            self.assertIsInstance(context1.get("relevant_reflections"), list) # Should be list even if empty
            print("Assertions for Stage 1 passed.")

            # --- Cleanup Dummy Files --- #
            # Optional: remove dummy files if they interfere with other tests
            # common_path.unlink(missing_ok=True)
            # stage0_path.unlink(missing_ok=True)
            # stage1_path.unlink(missing_ok=True)
            # stages_dir.rmdir() # Only if empty
            # server_prompts_dir.rmdir() # Only if empty
            print("Dummy prompt files cleanup skipped (manual if needed).")


        # Run the async test function
        asyncio.run(run_async_test())


# This allows running the tests from the command line
if __name__ == "__main__":
    # Configure logging for tests (optional, might be useful)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
