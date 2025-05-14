import unittest
import os
import shutil
from pathlib import Path
import sys
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock, ANY
import chromadb
import logging
import json
import yaml
import pytest

# <<< MOVE CHUNGOIDENGINE IMPORT OUTSIDE TRY/EXCEPT >>>
from chungoid.engine import ChungoidEngine

# Try importing other necessary components
try:
    from chungoid.utils.state_manager import StateManager, StatusFileError, ChromaOperationError
    from chungoid.schemas.common_enums import StageStatus
    from chungoid.utils import chroma_utils
except ImportError as e:
    print(f"Failed to import necessary components: {e}")
    pass

class TestIntegration(unittest.TestCase):
    TEST_DIR = Path("./test_project_integration")
    CHUNGOID_DIR = TEST_DIR / ".chungoid"
    STATUS_FILE = CHUNGOID_DIR / "project_status.json"
    logger = logging.getLogger(__name__)

    def setUp(self):
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)
        self.TEST_DIR.mkdir()
        # Ensure parent of TEST_DIR is added to sys.path if needed for imports during engine init?
        # Typically pytest handles this, but double-check if engine init fails.

    def tearDown(self):
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)

    def test_01_initialize_and_get_status(self):
        """Test initializing a project and then getting its status via engine."""
        print(f"\nRunning test: test_01_initialize_and_get_status in {os.getcwd()}")

        # Define an async inner function (might not be needed if all calls are sync)
        # Keeping for consistency for now.
        async def run_async_test():
            # <<< CHANGE START >>>
            # Instantiate the engine
            print(f"Instantiating ChungoidEngine for: {self.TEST_DIR.resolve()}")
            try:
                engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
            except Exception as e:
                self.fail(f"ChungoidEngine failed to initialize: {e}")
            print("Engine instantiated.")

            # Call initialize_project tool via engine
            print("Executing initialize_project tool...")
            init_result = engine.execute_mcp_tool(
                tool_name="initialize_project",
                tool_arguments={}, # Uses engine's context
                tool_call_id="init-call-1"
            )
            print(f"Initialization result: {init_result}")
            self.assertTrue(
                self.CHUNGOID_DIR.exists(), msg="Chungoid directory should exist after init"
            )
            self.assertTrue(self.STATUS_FILE.exists(), msg="Status file should exist after init")
            # Check new result structure
            self.assertIsInstance(init_result.get("content"), list)
            self.assertEqual(init_result["content"][0]["type"], "text")
            self.assertIn("Project initialized at", init_result["content"][0]["text"])
            self.assertEqual(init_result.get("toolCallId"), "init-call-1")

            # Call get_project_status tool via engine
            print("Executing get_project_status tool...")
            status_result = engine.execute_mcp_tool(
                tool_name="get_project_status",
                tool_arguments={},
                tool_call_id="status-call-1"
            )
            print(f"Status result: {status_result}")
            # Check new result structure (assuming it returns the status dict directly)
            self.assertEqual(status_result.get("toolCallId"), "status-call-1")
            self.assertIsInstance(status_result.get("content"), list)
            self.assertEqual(len(status_result["content"]), 1)
            self.assertEqual(status_result["content"][0]["type"], "text")
            status_json_string = status_result["content"][0]["text"]
            try:
                status_data = json.loads(status_json_string)
            except json.JSONDecodeError:
                self.fail(f"Failed to decode JSON from get_project_status result: {status_json_string}")
            self.assertIsInstance(status_data.get("runs"), list, msg="Parsed status data should have a 'runs' list")
            # <<< CHANGE END >>>
            print("Initialize and Get Status test passed.")

        # Run the async function using asyncio.run()
        asyncio.run(run_async_test())

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
            with patch('chungoid.utils.chroma_utils.get_chroma_client', return_value=test_client):

                # Test get_or_create_collection
                # <<< Assuming get_or_create_collection itself is now ASYNC >>>
                def run_create():
                    collection = chroma_utils.get_or_create_collection(collection_name)
                    self.assertIsNotNone(collection, "Failed to get/create collection")
                    self.assertEqual(collection.name, collection_name)
                run_create()

                # Test add_documents (assuming it's async)
                docs = ["doc1", "doc2"]
                ids = ["id1", "id2"]
                metadatas = [{"type": "test"}, {"type": "test"}]
                def run_add():
                    success = chroma_utils.add_documents(collection_name, docs, metadatas, ids)
                    self.assertTrue(success, "Failed to add documents")
                run_add()

                # Test query_documents (assuming it's async)
                def run_query():
                    results = chroma_utils.query_documents(collection_name, query_texts=["doc1"], n_results=1)
                    self.assertIsNotNone(results)
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0]["id"], "id1")
                run_query()

        finally:
            # Clean up temp ChromaDB directory
            shutil.rmtree(temp_chroma_dir, ignore_errors=True)
            print("Cleaned up temp dir: {self.TEST_DIR.resolve()}")

    # --- Tests for Reflection Loading/Retrieval Error Handling via Engine ---

    def test_08_engine_load_reflections_query_chroma_unavailable(self):
        """Test engine's 'load_reflections' tool when ChromaDB query operation fails."""
        print(f"\nRunning test: test_08_engine_load_reflections_query_chroma_unavailable in {os.getcwd()}")

        # Instantiate the real engine
        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            # Configure the mock instance's method
            mock_sm_instance.get_reflection_context_from_chroma.side_effect = ChromaOperationError("ChromaDB query client is not available.")
            
            async def run_async_test():
                print("Attempting to load reflections via engine with mocked state_manager instance...")
                load_result = engine.execute_mcp_tool(
                    tool_name="load_reflections",
                    tool_arguments={"query_text": "test query", "n_results": 1},
                    tool_call_id="load-reflect-fail-08"
                )
                print(f"Load reflections tool result: {load_result}")

                # Assert the method call on the mocked instance
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(query="test query", n_results=1)

                # Assert the error structure
                self.assertIsInstance(load_result.get("error"), dict)
                self.assertIn("ChromaDB query client is not available.", load_result["error"].get("message", ""))
                self.assertEqual(load_result["error"].get("code"), -32001) # Generic tool execution error
                self.assertEqual(load_result.get("toolCallId"), "load-reflect-fail-08")

            asyncio.run(run_async_test())


    def test_09_engine_load_reflections_query_success_empty(self):
        """Test engine's 'load_reflections' tool success when ChromaDB query returns no reflections."""
        print(f"\nRunning test: test_09_engine_load_reflections_query_success_empty in {os.getcwd()}")
        
        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            mock_sm_instance.get_reflection_context_from_chroma.return_value = [] # Direct return_value, not AsyncMock

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))
                print("Mock project initialized for engine.")

                print("Calling load_reflections tool via engine (expecting empty result)...")
                load_result = engine.execute_mcp_tool(
                    tool_name="load_reflections",
                    tool_arguments={"query_text": "empty query", "n_results": 5},
                    tool_call_id="load-reflect-empty-09"
                )
                print(f"Load reflections tool result: {load_result}")

                # mock_sm_instance = MockStateManagerClass.return_value # No longer need this
                # MockStateManagerClass.assert_called_once_with(target_directory=str(self.TEST_DIR.resolve()), server_stages_dir=ANY, chroma_client=ANY) # Not applicable with patch.object
                
                # <<< CHANGE query_text to query in assertion >>>
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(
                    query="empty query", n_results=5 # Use 'query' here
                )

                self.assertIsInstance(load_result.get("content"), list)
                self.assertEqual(len(load_result["content"]), 1)
                self.assertEqual(load_result["content"][0]["type"], "text")
                self.assertEqual(load_result["content"][0]["text"], "[]") # Expecting JSON string of empty list
                self.assertEqual(load_result.get("toolCallId"), "load-reflect-empty-09")
                print("Load reflections tool succeeded with empty data.")

            asyncio.run(run_async_test())


    def test_10_engine_load_reflections_query_success_with_data(self):
        """Test engine's 'load_reflections' tool success when ChromaDB query returns reflections."""
        print(f"\nRunning test: test_10_engine_load_reflections_query_success_with_data in {os.getcwd()}")

        sample_reflections_from_query = [
            {'id': 'uuid1', 'document': 'Queried Reflection 1', 'metadata': {'stage': 1.0}},
            {'id': 'uuid2', 'document': 'Queried Reflection 2', 'metadata': {'stage': 0.0}}
        ]
        
        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            # <<< Mock return value directly (it's a sync method) >>>
            mock_sm_instance.get_reflection_context_from_chroma.return_value = sample_reflections_from_query

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))
                print("Mock project initialized for engine.")

                print("Calling load_reflections tool via engine (expecting data)...")
                load_result = engine.execute_mcp_tool(
                    tool_name="load_reflections",
                    tool_arguments={"query_text": "data query", "n_results": 2},
                    tool_call_id="load-reflect-data-10"
                )
                print(f"Load reflections tool result: {load_result}")

                # MockStateManagerClass.assert_called_once_with(...) # Remove old assertion
                
                # <<< CHANGE query_text to query in assertion >>>
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(query="data query", n_results=2)
                
                self.assertIsInstance(load_result.get("content"), list)
                self.assertEqual(len(load_result["content"]), 1)
                self.assertEqual(load_result["content"][0]["type"], "text")
                
                returned_json_str = load_result["content"][0]["text"]
                try:
                    returned_data = json.loads(returned_json_str)
                except json.JSONDecodeError:
                    self.fail(f"Failed to decode JSON from load_reflections result: {returned_json_str}")
                
                self.assertEqual(returned_data, sample_reflections_from_query)
                self.assertEqual(load_result.get("toolCallId"), "load-reflect-data-10")
                print("Load reflections tool succeeded with data.")

            asyncio.run(run_async_test())

    # Tests 11, 12, 13 for 'retrieve_reflections' are functionally similar to 08, 09, 10
    # now that 'load_reflections' tool in engine handles querying. Renaming for clarity.

    def test_11_engine_retrieve_reflections_chroma_unavailable(self):
        """Test engine's 'load_reflections' (as retrieval) when ChromaDB query fails."""
        print(f"\nRunning test: test_11_engine_retrieve_reflections_chroma_unavailable in {os.getcwd()}")
        
        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            mock_sm_instance.get_reflection_context_from_chroma.side_effect = ChromaOperationError("ChromaDB query failed for retrieval.")

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))
                
                print("Attempting to retrieve reflections via engine tool with mocked unavailable Chroma client...")
                retrieve_result = engine.execute_mcp_tool(
                    tool_name="load_reflections", # Using 'load_reflections' as the query tool
                    tool_arguments={"query_text": "some query for retrieval", "n_results": 3},
                    tool_call_id="retrieve-fail-11"
                )
                print(f"Retrieve reflections tool result: {retrieve_result}")

                # MockStateManagerClass.assert_called_once_with(...) # Remove old assertion
                
                # <<< Check call with query= >>>
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(query="some query for retrieval", n_results=3)

                self.assertIsInstance(retrieve_result.get("error"), dict)
                self.assertIn("ChromaDB query failed for retrieval.", retrieve_result["error"].get("message", ""))
                self.assertEqual(retrieve_result["error"].get("code"), -32001)
                self.assertEqual(retrieve_result.get("toolCallId"), "retrieve-fail-11")

            asyncio.run(run_async_test())

    def test_12_engine_retrieve_reflections_success_empty(self):
        """Test engine's 'load_reflections' (as retrieval) success when ChromaDB query returns no results."""
        print(f"\nRunning test: test_12_engine_retrieve_reflections_success_empty in {os.getcwd()}")

        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            mock_sm_instance.get_reflection_context_from_chroma.return_value = []

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))

                print("Calling retrieve_reflections via engine tool (empty result)...")
                retrieve_result = engine.execute_mcp_tool(
                    tool_name="load_reflections",
                    tool_arguments={"query_text": "empty retrieval query", "n_results": 5},
                    tool_call_id="retrieve-empty-12"
                )
                print(f"Retrieve reflections tool result: {retrieve_result}")

                # MockStateManagerClass.assert_called_once_with(...) # Remove old assertion
                
                # <<< Check call with query= >>>
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(
                    query="empty retrieval query", n_results=5
                )

                self.assertIsInstance(retrieve_result.get("content"), list)
                self.assertEqual(retrieve_result["content"][0]["text"], "[]")
                self.assertEqual(retrieve_result.get("toolCallId"), "retrieve-empty-12")
                print("Retrieve reflections via engine tool succeeded with empty data.")

            asyncio.run(run_async_test())

    def test_13_engine_retrieve_reflections_success_with_data(self):
        """Test engine's 'load_reflections' (as retrieval) success when ChromaDB query returns results."""
        print(f"\nRunning test: test_13_engine_retrieve_reflections_success_with_data in {os.getcwd()}")

        sample_results_for_retrieval = [
            {"id": "doc_r1", "document": "Relevant retrieved reflection 1", "metadata": {"stage": 1}},
            {"id": "doc_r2", "document": "Relevant retrieved reflection 2", "metadata": {"stage": 2}},
        ]
        
        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            mock_sm_instance.get_reflection_context_from_chroma.return_value = sample_results_for_retrieval

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))

                print("Calling retrieve_reflections via engine tool (with data)...")
                query = "data retrieval query"
                num_results = 2
                retrieve_result = engine.execute_mcp_tool(
                    tool_name="load_reflections",
                    tool_arguments={"query_text": query, "n_results": num_results},
                    tool_call_id="retrieve-data-13"
                )
                print(f"Retrieve reflections tool result: {retrieve_result}")

                # MockStateManagerClass.assert_called_once_with(...) # Remove old assertion
                
                # <<< Check call with query= >>>
                mock_sm_instance.get_reflection_context_from_chroma.assert_called_once_with(
                    query=query, n_results=num_results
                )

                self.assertIsInstance(retrieve_result.get("content"), list)
                returned_json_str = retrieve_result["content"][0]["text"]
                try:
                    returned_data = json.loads(returned_json_str)
                except json.JSONDecodeError:
                    self.fail(f"Failed to decode JSON from retrieve_result: {returned_json_str}")
                self.assertEqual(returned_data, sample_results_for_retrieval)
                self.assertEqual(retrieve_result.get("toolCallId"), "retrieve-data-13")
                print("Retrieve reflections via engine tool succeeded with data.")

            asyncio.run(run_async_test())

    # --- Tests for handle_submit_stage_artifacts via Engine --- #

    # @patch('chungoid.utils.state_manager.StateManager') # REMOVED
    def test_14_engine_submit_artifacts_update_status_fails(self): # <<< RENAME TEST & REMOVE ARG
        """Test engine's 'submit_stage_artifacts' tool when the underlying state_manager.update_status call fails."""
        print(f"\nRunning test: test_14_engine_submit_artifacts_update_status_fails in {os.getcwd()}")

        engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
        # <<< USE PATCH.OBJECT ON THE ENGINE'S state_manager INSTANCE >>>
        with patch.object(engine, 'state_manager', autospec=True) as mock_sm_instance:
            # <<< Mock update_status to return False >>>
            mock_sm_instance.update_status.return_value = False
            # persist_reflections_to_chroma is NOT called by the wrapper, so no need to mock it here.

            async def run_async_test():
                if not self.CHUNGOID_DIR.exists(): self.CHUNGOID_DIR.mkdir(parents=True)
                if not self.STATUS_FILE.exists(): self.STATUS_FILE.write_text(json.dumps({"runs": []}))
                print("Mock project initialized for engine.")

                artifact_rel_path = "output/stage1_output.txt"
                artifact_full_path = self.TEST_DIR / artifact_rel_path
                artifact_full_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_full_path.write_text("Stage 1 output content")

                print("Submitting artifacts via engine (expecting update_status to fail)...")
                submit_result = engine.execute_mcp_tool(
                    tool_name="submit_stage_artifacts",
                    tool_arguments={
                        "stage_number": 1.0,
                        "stage_result_status": "PASS",
                        "generated_artifacts": {artifact_rel_path: "Stage 1 output content"},
                        "reflection_text": "This is a reflection text for stage 1."
                    },
                    tool_call_id="submit-artifact-fail-14"
                )
                print(f"Submit artifacts tool result: {submit_result}")

                # Check that update_status was called correctly
                mock_sm_instance.update_status.assert_called_once_with(
                    stage=1.0,
                    status="PASS",
                    artifacts=[artifact_rel_path],
                    reflection_text="This is a reflection text for stage 1."
                    # error_details is not passed in this call
                )
                
                # Check that persist_reflections_to_chroma was NOT called by this flow
                mock_sm_instance.persist_reflections_to_chroma.assert_not_called()

                # Assert the error structure from the caught RuntimeError
                self.assertIsInstance(submit_result.get("error"), dict)
                self.assertIn("Failed to update status for stage 1.0", submit_result["error"].get("message", "")) # Error from wrapper
                self.assertEqual(submit_result["error"].get("code"), -32001) # Generic tool execution error
                self.assertEqual(submit_result.get("toolCallId"), "submit-artifact-fail-14")

            asyncio.run(run_async_test())

    # --- Tests for handle_get_file --- #

    def test_15_get_file(self):
        """Test the get_file tool handler for success and failure cases via engine."""
        print(f"\nRunning test: test_15_get_file in {os.getcwd()}")

        async def run_async_test(): # Keep async for potential future async setup/teardown
            # <<< CHANGE START >>>
            # Instantiate the engine
            print(f"Instantiating ChungoidEngine for: {self.TEST_DIR.resolve()}")
            try:
                engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
            except Exception as e:
                self.fail(f"ChungoidEngine failed to initialize: {e}")
            print("Engine instantiated.")

            # Initialize project using the engine tool
            print("Initializing project via engine...")
            init_result = engine.execute_mcp_tool(
                tool_name="initialize_project",
                tool_arguments={},
                tool_call_id="init-call-15"
            )
            # Basic check for init success (detailed check in test_01)
            self.assertIsInstance(init_result.get("content"), list)
            print("Project initialized.")
            # <<< CHANGE END >>>

            # Create a dummy file
            docs_dir = self.TEST_DIR / "docs"
            docs_dir.mkdir(exist_ok=True)
            test_file_path = docs_dir / "myfile.txt"
            test_content = "This is the content of the test file.\nLine 2."
            with open(test_file_path, "w") as f:
                f.write(test_content)
            print(f"Created test file: {test_file_path}")

            # --- Test Success Case ---
            print("Testing get_file success case via engine...")
            # <<< CHANGE START >>>
            success_result = engine.execute_mcp_tool(
                tool_name="get_file",
                tool_arguments={"relative_path": "docs/myfile.txt"},
                tool_call_id="get-file-success"
            )
            # Check result structure
            self.assertIsInstance(success_result.get("content"), list)
            self.assertEqual(len(success_result["content"]), 1)
            self.assertEqual(success_result["content"][0]["type"], "text")
            # Adjust assertion to expect the prepended string
            expected_success_text = f"Content of {test_file_path.resolve()}:\\n\\n{test_content}"
            self.assertEqual(
                success_result["content"][0]["text"],
                expected_success_text,
                msg="File content does not match expected format",
            )
            self.assertEqual(success_result.get("toolCallId"), "get-file-success")
            # <<< CHANGE END >>>
            print("Get file success case passed.")

            # --- Test Not Found Case ---
            print("Testing get_file not found case via engine...")
            # <<< CHANGE START >>>
            not_found_result = engine.execute_mcp_tool(
                tool_name="get_file",
                tool_arguments={"relative_path": "docs/nosuchfile.txt"},
                tool_call_id="get-file-notfound"
            )
            # Check error structure (execute_mcp_tool should catch and format)
            self.assertIsInstance(not_found_result.get("error"), dict)
            self.assertIn("File not found", not_found_result["error"].get("message", ""))
            self.assertEqual(not_found_result["error"].get("code"), -32001) # Generic tool execution error
            self.assertEqual(not_found_result.get("toolCallId"), "get-file-notfound")
            # <<< CHANGE END >>>
            print("Not found case passed.")

            # --- Test Access Denied Case (Simulated via path traversal) ---
            print("Testing get_file access denied/invalid path case via engine...")
            # <<< ADD START >>>
            # Create dummy file outside project dir for traversal test
            outside_file_path = self.TEST_DIR.parent / "outside.txt"
            outside_file_content = "This content is outside the project."
            with open(outside_file_path, "w") as f:
                f.write(outside_file_content)
            print(f"Created temporary outside file: {outside_file_path}")
            # <<< ADD END >>>

            # <<< CHANGE START >>>
            traversal_result = engine.execute_mcp_tool(
                tool_name="get_file",
                tool_arguments={"relative_path": "../outside.txt"},
                tool_call_id="get-file-traversal"
            )
            # Check error structure
            self.assertIsInstance(traversal_result.get("error"), dict)
            self.assertIn("is outside the project directory", traversal_result["error"].get("message", ""))
            self.assertEqual(traversal_result["error"].get("code"), -32001)
            self.assertEqual(traversal_result.get("toolCallId"), "get-file-traversal")
            # <<< CHANGE END >>>
            
            # <<< ADD START >>>
            # Clean up dummy outside file
            if outside_file_path.exists():
                outside_file_path.unlink()
                print(f"Cleaned up temporary outside file: {outside_file_path}")
            # <<< ADD END >>>
            print("Get file path traversal case passed.")

        asyncio.run(run_async_test())

    def test_16_prepare_next_stage(self):
        """Test the prepare_next_stage tool handler via ChungoidEngine."""
        print(f"\nRunning test: test_16_prepare_next_stage in {os.getcwd()}")

        async def run_async_test():
            # Helper to create dummy prompt files (remains the same, ensure it's called if needed by engine)
            def _create_dummy_prompt_files_inner():
                core_root_helper = Path(__file__).parent.parent 
                server_prompts_path_helper = core_root_helper / "server_prompts"
                stages_path_helper = server_prompts_path_helper / "stages"
                stages_path_helper.mkdir(parents=True, exist_ok=True)
                common_path_helper = server_prompts_path_helper / "common.yaml"
                print(f"DEBUG TEST (_create_dummy_prompt_files_inner): server_prompts_path_helper abs: {server_prompts_path_helper.resolve()}")
                # ... (rest of _create_dummy_prompt_files_inner is unchanged) ...
                # Stage Files (Must have system_prompt and user_prompt strings)
                for i in range(6):
                    stage_file_helper = stages_path_helper / f"stage{i}_helper.yaml" 
                    stage_content_helper = (
                        f"description: 'Dummy Stage Helper {i}'\n"
                        f"system_prompt: 'Sys Prompt Helper {i}'\n"
                        f"user_prompt: 'User Prompt Helper {i}'\n"
                    )
                    with open(stage_file_helper, 'w') as f:
                        f.write(stage_content_helper)
                print(f"DEBUG TEST: Created dummy helper prompt files in {stages_path_helper.parent.resolve()}")

            # --- Engine Setup --- #
            print(f"DEBUG TEST: Instantiating ChungoidEngine for project: {self.TEST_DIR.resolve()}")
            try:
                engine = ChungoidEngine(str(self.TEST_DIR.resolve()))
            except Exception as e:
                self.fail(f"ChungoidEngine failed to initialize for test_16: {e}")
            print("DEBUG TEST: ChungoidEngine instantiated for test_16.")

            # --- Setup: Initialize Project via Engine --- #
            print(f"DEBUG TEST: Initializing project in: {self.TEST_DIR.resolve()} via engine tool")
            init_result = engine.execute_mcp_tool(
                tool_name="initialize_project",
                tool_arguments={},
                tool_call_id="test16-init"
            )
            self.assertNotIn("error", init_result, msg=f"Initialization failed: {init_result.get('error')}")
            self.assertEqual(init_result.get("toolCallId"), "test16-init")
            # Add more assertions if needed, e.g., from test_01
            self.assertTrue(self.CHUNGOID_DIR.exists(), msg="Chungoid directory should exist after init via engine")
            self.assertTrue(self.STATUS_FILE.exists(), msg="Status file should exist after init via engine")
            print("DEBUG TEST: Project initialized via engine.")

            # --- Setup: Create Dummy Stage/Common Files --- #
            # These are the primary files the engine should find due to its own init path logic.
            # No need to call _create_dummy_prompt_files_inner() if the main files are correctly placed
            # for the *engine's* PromptManager (which uses chungoid-core/server_prompts).
            # However, if test_16 specifically needs to test loading of *these exact dummy files created by the test*,
            # then the engine's PromptManager needs to be pointed to them or they need to be placed where
            # the standard PromptManager looks (i.e., in the main chungoid-core/server_prompts).
            # For now, let's assume the engine uses its default server_prompts and we want to test that.
            # The `test_16_prepare_next_stage` name implies testing the tool, not custom prompt loading.
            
            # The print statements for path creation in the test can be kept for sanity checking but 
            # the files created by the test at `chungoid-core/server_prompts` are the ones 
            # the *engine's default PromptManager* will use.
            # core_root_main = Path(__file__).parent.parent 
            # server_prompts_dir_main = core_root_main / 'server_prompts'
            # stages_dir_main = server_prompts_dir_main / 'stages'
            # stages_dir_main.mkdir(parents=True, exist_ok=True) # Ensure it exists
            # common_path_main = server_prompts_dir_main / 'common.yaml'
            # stage0_path_main = stages_dir_main / 'stage0.yaml'
            # stage1_path_main = stages_dir_main / 'stage1.yaml'

            # print(f"DEBUG TEST (main setup for engine): server_prompts_dir_main abs: {server_prompts_dir_main.resolve()}")
            # print(f"DEBUG TEST (main setup for engine): stages_dir_main abs: {stages_dir_main.resolve()}")
            # print(f"DEBUG TEST (main setup for engine): common_path_main abs: {common_path_main.resolve()}")

            # # Ensure these files have minimal valid content for PromptManager - REMOVING THIS BLOCK
            # # The engine should use the globally available server_prompts.
            # if not common_path_main.exists() or common_path_main.read_text().strip() == "":
            #     common_path_main.write_text("preamble: COMMON PREAMBLE MAIN FROM TEST_16\npostamble: COMMON POSTAMBLE MAIN FROM TEST_16")
            #     print(f"DEBUG TEST: Wrote default content to {common_path_main.resolve()}")
            # 
            # if not stage0_path_main.exists() or stage0_path_main.read_text().strip() == "":
            #     stage0_path_main.write_text("description: 'Stage 0 Main from Test_16'\nprompt_details: Stage 0 Details {{ context_data.initial_goal | default('') }}\nsystem_prompt: 'Sys Prompt Main 0 from Test_16'\nuser_prompt: User prompt for Stage Main 0 from Test_16.")
            #     print(f"DEBUG TEST: Wrote default content to {stage0_path_main.resolve()}")
            # 
            # if not stage1_path_main.exists() or stage1_path_main.read_text().strip() == "":
            #     stage1_path_main.write_text("description: 'Stage 1 Main from Test_16'\nprompt_details: Stage 1 Details\nsystem_prompt: 'Sys Prompt Main 1 from Test_16'\nuser_prompt: User prompt for Stage Main 1 from Test_16. Last status was {{ context_data.last_status.status }}")
            #     print(f"DEBUG TEST: Wrote default content to {stage1_path_main.resolve()}")

            # print(f"DEBUG TEST: Ensured main dummy prompt files exist in {server_prompts_dir_main.resolve()} for engine use.")
            print(f"DEBUG TEST: Engine will use globally available prompt files from chungoid-core/server_prompts/")

            # --- Test Initial Call (Should prepare Stage 0) --- #
            print("DEBUG TEST: Calling prepare_next_stage tool via engine for the first time...")
            prepare_result_1 = engine.execute_mcp_tool(
                tool_name="prepare_next_stage",
                tool_arguments={},
                tool_call_id="test16-prepare1"
            )
            print(f"DEBUG TEST: First prepare_next_stage result: {prepare_result_1}")
            self.assertNotIn("error", prepare_result_1, msg=f"First prepare_next_stage failed: {prepare_result_1.get('error')}")
            self.assertEqual(prepare_result_1.get("toolCallId"), "test16-prepare1")
            self.assertIsInstance(prepare_result_1.get("content"), list)
            content_dict_1 = yaml.safe_load(prepare_result_1["content"][0]["text"]) # Assuming content is YAML string
            self.assertEqual(content_dict_1.get("status"), "success")
            self.assertEqual(content_dict_1.get("next_stage"), 0.0)
            self.assertIn("User prompt for Stage 0.", content_dict_1.get("prompt", "")) # Use default prompt content
            self.assertIn("COMMON PREAMBLE", content_dict_1.get("prompt", ""))
            self.assertIn("COMMON POSTAMBLE", content_dict_1.get("prompt", ""))

            # --- Simulate Stage 0 Completion and Submit Artifacts via Engine --- # 
            print("DEBUG TEST: Simulating Stage 0 completion and submitting artifacts via engine...")
            submit_args_0 = {
                "stage_number": 0.0,
                "stage_result_status": "PASS",
                "generated_artifacts": {"docs/stage0_output.txt": "Output from stage 0"},
                "reflection_text": "Reflection for stage 0 from test_16"
            }
            submit_result_0 = engine.execute_mcp_tool(
                tool_name="submit_stage_artifacts",
                tool_arguments=submit_args_0,
                tool_call_id="test16-submit0"
            )
            print(f"DEBUG TEST: Submit artifacts for stage 0 result: {submit_result_0}")
            self.assertNotIn("error", submit_result_0, msg=f"Submit artifacts for stage 0 failed: {submit_result_0.get('error')}")
            self.assertEqual(submit_result_0.get("toolCallId"), "test16-submit0")
            self.assertIn("Stage 0.0 submitted with status PASS and 1 artifacts.", submit_result_0["content"][0]["text"]) # Corrected assertion

            # --- Test Second Call (Should prepare Stage 1) --- #
            print("DEBUG TEST: Calling prepare_next_stage tool via engine for the second time...")
            prepare_result_2 = engine.execute_mcp_tool(
                tool_name="prepare_next_stage",
                tool_arguments={},
                tool_call_id="test16-prepare2"
            )
            print(f"DEBUG TEST: Second prepare_next_stage result: {prepare_result_2}")
            self.assertNotIn("error", prepare_result_2, msg=f"Second prepare_next_stage failed: {prepare_result_2.get('error')}")
            self.assertEqual(prepare_result_2.get("toolCallId"), "test16-prepare2")
            content_dict_2 = yaml.safe_load(prepare_result_2["content"][0]["text"])
            self.assertEqual(content_dict_2.get("status"), "success")
            self.assertEqual(content_dict_2.get("next_stage"), 1.0)
            self.assertIn("User prompt for Stage 1.", content_dict_2.get("prompt", "")) # Corrected assertion
            self.assertIn("Last status was PASS", content_dict_2.get("prompt", "")) # Check context injection
            self.assertIn("COMMON PREAMBLE", content_dict_2.get("prompt", ""))

            print("DEBUG TEST: test_16_prepare_next_stage completed successfully.")

        asyncio.run(run_async_test())


# This allows running the tests from the command line
if __name__ == "__main__":
    # Configure logging for tests (optional, might be useful)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
