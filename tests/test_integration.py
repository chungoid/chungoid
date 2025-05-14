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
import uuid
import copy

# <<< MOVE CHUNGOIDENGINE IMPORT OUTSIDE TRY/EXCEPT >>>
from chungoid.engine import ChungoidEngine
from chungoid.utils.llm_provider import MockLLMProvider
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider, DictAgentProvider, AgentCallable
from chungoid.runtime.agents.master_planner_agent import master_planner_agent_card, PROMPTS_DIR as MP_PROMPTS_DIR
from chungoid.utils import config_loader

# Try importing other necessary components
try:
    from chungoid.utils.state_manager import StateManager, StatusFileError, ChromaOperationError
    from chungoid.schemas.common_enums import StageStatus
    from chungoid.utils import chroma_utils
    from chungoid.schemas.user_goal_schemas import UserGoalRequest
    from chungoid.schemas.master_flow import MasterExecutionPlan
    from chungoid.utils.agent_registry import AgentCard, AGENT_REGISTRY
except ImportError as e:
    print(f"Failed to import necessary components: {e}")
    pass

class TestIntegration(unittest.TestCase):
    TEST_DIR = Path("./test_project_integration")
    CHUNGOID_DIR = TEST_DIR / ".chungoid"
    STATUS_FILE = CHUNGOID_DIR / "project_status.json"
    logger = logging.getLogger(__name__)

    @classmethod
    def setUpClass(cls):
        cls.test_project_name = "test_integration_project"
        cls.base_project_dir = Path(__file__).parent / "test_projects"
        cls.project_dir = cls.base_project_dir / cls.test_project_name
        
        # Initialize providers for use in tests
        cls.llm_provider = MockLLMProvider() # Central LLM provider
        
        # Agent provider setup - using DictAgentProvider for controlled test environment
        # cls.agent_registry_instance = DictAgentProvider({}) # Store cards here # <<< REMOVE
        # cls.agent_registry_instance.agents[master_planner_agent_card.agent_id] = master_planner_agent_card # Register master planner # <<< REMOVE
        # If MasterPlannerAgent needs to call other agents via its agent_provider, they also need to be registered here.
        # For now, assuming its internal agent selection logic might be mocked or uses hardcoded fallbacks if necessary during planning.

        # If the engine is created per-test, these will be passed. 
        # If there's a cls.engine, it needs to be initialized with these.
        # For this test, we'll instantiate engine directly in the test method.

        if cls.project_dir.exists():
            shutil.rmtree(cls.project_dir)
        cls.project_dir.mkdir(parents=True)

        # Override config to use local ChromaDB for this test class
        # cls.original_config = get_config() # Not needed if we store/restore module var
        cls.original_config_module_var = config_loader._config # Store current module-level config
        
        current_actual_config = config_loader.get_config() # Ensure it's loaded if it was None
        test_config_override = copy.deepcopy(current_actual_config) 
        # Modify test_config_override as needed for tests, e.g.:
        # test_config_override["chromadb"]["mode"] = "persistent"
        # test_config_override["chromadb"]["persist_path"] = str(cls.project_dir / ".test_integration_chroma_db")

        config_loader._config = test_config_override # Directly set the override

        # Ensure a unique path for ChromaDB for this test run if needed
        # test_config_override.chromadb_path = str(cls.project_dir / ".test_chroma_db") 
        # For integration, we often want it to behave as close to prod as possible,
        # so using the default from config or StateManager's logic might be fine if isolated.
        # Let's assume StateManager handles ChromaDB pathing correctly within project_dir.
        # set_config_override(test_config_override) # Removed as set_config_override doesn't exist

        # Initialize a basic ChungoidEngine for setup tasks if needed, or do it in tests
        # cls.engine = ChungoidEngine(str(cls.project_dir), cls.llm_provider, cls.agent_registry_instance)
        # cls.engine.execute_mcp_tool("initialize_project", {})

    def setUp(self):
        """Ensure each test starts with a clean, initialized project state if needed."""
        # Re-initialize engine for each test to ensure isolation of state and provider mocks
        # This is more robust than a class-level engine if tests modify LLM/agent provider state.
        self.current_llm_provider = MockLLMProvider() # Fresh for each test

        self.current_agent_provider = DictAgentProvider({}) # Fresh DictAgentProvider
        # self.current_agent_provider.agents[master_planner_agent_card.agent_id] = master_planner_agent_card # <<< REMOVE THIS LINE
        
        # Project directory for this specific test method
        self.TEST_DIR = self.base_project_dir / self.test_project_name / self._testMethodName
        
        self.engine = ChungoidEngine(
            str(self.project_dir), 
            llm_provider=self.current_llm_provider, 
            agent_provider=self.current_agent_provider
        )
        self.engine.execute_mcp_tool("initialize_project", {}) # Initialize project for each test
        self.engine.state_manager._pending_reflection_text = None # Clear pending reflections

    def tearDown(self):
        if self.TEST_DIR.exists():
            shutil.rmtree(self.TEST_DIR)

    @classmethod
    def tearDownClass(cls):
        # Restore the original config_loader._config that was cached at the start of setUpClass
        if hasattr(cls, 'original_config_module_var'):
            config_loader._config = cls.original_config_module_var
        
        # Clean up project directory created by setUpClass
        if hasattr(cls, 'project_dir') and cls.project_dir.exists():
            shutil.rmtree(cls.project_dir)
        
        # If there are other class-level cleanups, add them here.

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

    def test_30_create_master_plan_tool(self):
        """Test the create_master_plan tool via ChungoidEngine."""
        user_goal_id = f"test_goal_{uuid.uuid4()}"
        user_goal_request = UserGoalRequest(
            goal_id=user_goal_id,
            goal_description="Develop a weather app for Wonderland.",
            target_platform="mobile",
            key_constraints={"timeline": "short", "magic_level": "low"}
        )

        # --- Mock LLM Responses for MasterPlannerAgent steps ---
        # 1. Decomposition Prompt & Response
        # Construct path to prompts relative to this test file for robustness
        current_test_file_path = Path(__file__).resolve()
        # chungoid-core/tests/test_integration.py -> chungoid-core/tests/ -> chungoid-core/
        project_root_from_test = current_test_file_path.parent.parent 
        test_MP_PROMPTS_DIR = project_root_from_test / "server_prompts" / "master_planner"

        decomposition_prompt_template = (test_MP_PROMPTS_DIR / "decomposition_prompt.txt").read_text()
        expected_decomposition_prompt = decomposition_prompt_template.format(
            goal_description=user_goal_request.goal_description,
            target_platform=user_goal_request.target_platform,
            key_constraints=str(user_goal_request.key_constraints)
        ).strip()
        mock_decomposed_tasks_str = "1. Design UI for Alice\\n2. Implement Cheshire Cat API integration\\n3. Test with Mad Hatter"
        self.current_llm_provider.predefined_responses[expected_decomposition_prompt] = mock_decomposed_tasks_str

        # 2. Agent Selection Prompts & Responses (for each decomposed task)
        agent_selection_prompt_template = (test_MP_PROMPTS_DIR / "agent_selection_prompt.txt").read_text()
        decomposed_tasks_list = ["Design UI for Alice", "Implement Cheshire Cat API integration", "Test with Mad Hatter"]
        selected_agent_ids_for_tasks = ["ui_designer_wonderland", "api_cat_specialist", "mad_tester_general"]
        
        # This is the hardcoded agent details string from MasterPlannerAgent's fallback logic
        candidate_details_fallback = """- Agent ID: generic_task_agent
  Description: A generic agent capable of performing various tasks.
  Capabilities: Can execute general instructions.
  Expected Input Summary: text prompt
  Expected Output Summary: text result
- Agent ID: file_writer_agent
  Description: Writes content to a file.
  Capabilities: File I/O, content creation.
  Expected Input Summary: file_path, content
  Expected Output Summary: status_message"""

        for i, task_desc in enumerate(decomposed_tasks_list):
            expected_agent_selection_prompt = agent_selection_prompt_template.format(
                original_user_goal_description=user_goal_request.goal_description,
                current_decomposed_task_description=task_desc,
                candidate_agents_details_formatted=candidate_details_fallback 
            ).strip()
            mock_selection_response = json.dumps({
                "selected_agent_ids": [selected_agent_ids_for_tasks[i]],
                "justification": f"Agent {selected_agent_ids_for_tasks[i]} selected for {task_desc} due to expertise."
            })
            self.current_llm_provider.predefined_responses[expected_agent_selection_prompt] = mock_selection_response

        # 3. Sequencing Prompt & Response
        sequencing_prompt_template = (test_MP_PROMPTS_DIR / "sequencing_prompt.txt").read_text()
        # Construct tasks_for_prompt based on selected agents
        # MasterPlannerAgent assigns temporary IDs like T0, T1, T2 before calling _sequence_tasks
        tasks_for_sequencing_prompt_list = []
        for i, task_desc in enumerate(decomposed_tasks_list):
            tasks_for_sequencing_prompt_list.append(
                f"T{i}: {task_desc} (Assigned Agent: {selected_agent_ids_for_tasks[i]})"
            )
        tasks_for_sequencing_prompt_str = "\\n".join(tasks_for_sequencing_prompt_list)

        expected_sequencing_prompt = sequencing_prompt_template.format(
            goal_description=user_goal_request.goal_description,
            tasks_to_sequence=tasks_for_sequencing_prompt_str
        ).strip()
        # LLM returns them in order T0, T1, T2 for this test
        mock_sequenced_ids_str = "T0,T1,T2"
        self.current_llm_provider.predefined_responses[expected_sequencing_prompt] = mock_sequenced_ids_str

        # --- Mock state_manager.save_master_execution_plan ---
        with patch.object(self.engine.state_manager, 'save_master_execution_plan', return_value=True) as mock_save_plan:
            # --- Execute Tool ---
            # The tool returns the plan as a dict because it's JSON serialized for MCP
            tool_output_dict = self.engine.execute_mcp_tool(
                "create_master_plan", 
                {"user_goal": user_goal_request.model_dump()}
            )

            # --- Assertions ---
            self.assertIsInstance(tool_output_dict, dict)
            # The tool output is the MCP response structure, content is under "content"[0]["text"] as JSON string or the direct dict
            # Based on current engine.py, it returns plan.model_dump() directly if handler_sync doesn't wrap it.
            # The _create_master_plan_sync_wrapper returns plan.model_dump(), so tool_output_dict IS the plan dict.

            self.assertNotIn("error", tool_output_dict, f"Tool execution failed: {tool_output_dict.get('error')}")
            
            # Extract the actual plan JSON string from the MCP response structure
            self.assertIn("content", tool_output_dict)
            self.assertIsInstance(tool_output_dict["content"], list)
            self.assertGreater(len(tool_output_dict["content"]), 0)
            self.assertIn("text", tool_output_dict["content"][0])
            plan_json_string = tool_output_dict["content"][0]["text"]
            plan_dict_from_tool = json.loads(plan_json_string)

            # Parse back to MasterExecutionPlan
            try:
                generated_plan = MasterExecutionPlan(**plan_dict_from_tool)
            except Exception as e:
                self.fail(f"Failed to parse tool output into MasterExecutionPlan: {e}\nOutput: {tool_output_dict}")

            self.assertTrue(generated_plan.id.startswith(f"mep_{user_goal_id}"))
            # MasterPlannerAgent._format_plan uses goal_description[:50]
            expected_plan_name = f"Plan for: {user_goal_request.goal_description[:50]}..."
            self.assertEqual(generated_plan.name, expected_plan_name)
            self.assertEqual(generated_plan.original_request, user_goal_request)
            self.assertEqual(len(generated_plan.stages), 3)
            
            # MasterPlannerAgent uses temp_task_id (T0, T1, T2) as stage_id in the plan
            self.assertEqual(generated_plan.start_stage, "T0")

            # Stage T0 (Design UI for Alice)
            stage_t0 = generated_plan.stages.get("T0")
            self.assertIsNotNone(stage_t0)
            self.assertEqual(stage_t0.name, f"Stage 0: {decomposed_tasks_list[0][:100]}")
            self.assertEqual(stage_t0.agent_id, selected_agent_ids_for_tasks[0])
            self.assertEqual(stage_t0.number, 0.0)
            self.assertEqual(stage_t0.next_stage, "T1")
            self.assertIn("user_goal_description", stage_t0.inputs)
            self.assertEqual(stage_t0.inputs["task_description"], decomposed_tasks_list[0])
            self.assertNotIn("previous_stage_outputs", stage_t0.inputs)

            # Stage T1 (Implement Cheshire Cat API integration)
            stage_t1 = generated_plan.stages.get("T1")
            self.assertIsNotNone(stage_t1)
            self.assertEqual(stage_t1.name, f"Stage 1: {decomposed_tasks_list[1][:100]}")
            self.assertEqual(stage_t1.agent_id, selected_agent_ids_for_tasks[1])
            self.assertEqual(stage_t1.number, 1.0)
            self.assertEqual(stage_t1.next_stage, "T2")
            self.assertEqual(stage_t1.inputs["task_description"], decomposed_tasks_list[1])
            self.assertEqual(stage_t1.inputs.get("previous_stage_outputs"), "T0")

            # Stage T2 (Test with Mad Hatter)
            stage_t2 = generated_plan.stages.get("T2")
            self.assertIsNotNone(stage_t2)
            self.assertEqual(stage_t2.name, f"Stage 2: {decomposed_tasks_list[2][:100]}")
            self.assertEqual(stage_t2.agent_id, selected_agent_ids_for_tasks[2])
            self.assertEqual(stage_t2.number, 2.0)
            self.assertIsNone(stage_t2.next_stage) # next_stage is None for the final step
            self.assertEqual(stage_t2.inputs.get("previous_stage_outputs"), "T1")

            # Check if save was called with a MasterExecutionPlan object
            mock_save_plan.assert_called_once()
            args, _ = mock_save_plan.call_args
            saved_plan_arg = args[0]
            self.assertIsInstance(saved_plan_arg, MasterExecutionPlan)
            self.assertEqual(saved_plan_arg.id, generated_plan.id) # Verify it's the same plan
            self.assertEqual(saved_plan_arg.original_request.goal_id, user_goal_id)


# This allows running the tests from the command line
if __name__ == "__main__":
    # Configure logging for tests (optional, might be useful)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    unittest.main()
