import pytest
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import tempfile
import re
import shutil

from chungoid.schemas.orchestration import SharedContext
from chungoid.schemas.project_status_schema import ProjectStateV2, ArtifactDetails
from chungoid.runtime.services.context_resolution_service import ContextResolutionService # type: ignore # VSCode marks this as an error but it is fine

# Configure a logger for tests if needed, or rely on service's internal logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG) # Or another appropriate level for tests
# handler = logging.StreamHandler()
# logger.addHandler(handler)

class TestContextResolutionService:
    """
    Unit tests for the ContextResolutionService.
    """
    
    service: ContextResolutionService
    temp_dir: str
    project_root_path: Path
    project_status: ProjectStateV2
    shared_context: SharedContext

    def setup_method(self, method: Any) -> None:
        """Setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        self.service = ContextResolutionService()
        self.temp_dir = tempfile.mkdtemp()
        self.project_root_path = Path(self.temp_dir) / "test_project"
        self.project_root_path.mkdir(parents=True, exist_ok=True)

        # Initialize ProjectStateV2 (which serves as ProjectStatus)
        self.project_status = ProjectStateV2(
            project_id="test_proj", 
            initial_user_goal_summary="Test goal"
            # artifacts dict is default_factory=dict
        )

        self.shared_context = SharedContext(
            run_id="test_run_id",             # ADDED
            flow_id="test_flow_id",           # ADDED
            data={                            # MODIFIED: Nest initial data
                "project_id": "test_proj",
                "project_root_path": str(self.project_root_path),
                "project_status": self.project_status,
                "global_project_settings": {"some_global_setting": "global_value"},
                "outputs": {}, # Initialize as empty dict within data
                "initial_inputs": {
                    "initial_param": "initial_value"
                },
                "previous_stage_outputs": {} # Initialize as empty here too
            }
        )
        self.service.shared_context = self.shared_context # Assign to service instance

        # Example of adding an artifact for tests that need it
        # MODIFIED: update self.shared_context.data["outputs"]
        self.shared_context.data["outputs"] = {
            "stage_one": {"keyA": "valueA", "nested_dict": {"deep_key": "deep_val"}},
            "stage_two": {"keyB": [100, 200, {"sub_key": "sub_val"}]},
            "stage_with_hyphen_key": {"my-actual-key": "hyphen_value"}
        }
        # Artifacts should be added to project_status.artifacts, which is already in shared_context.data
        # MODIFIED: access project_status via self.shared_context.data
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts = { 
                "data_artifact": ArtifactDetails(artifact_id="data_artifact", name="data_artifact", path_on_disk="/tmp/artifacts/data.txt", metadata={"source": "upload", "size_kb": 2}),
                "code_artifact": ArtifactDetails(artifact_id="code_artifact", name="code_artifact", path_on_disk="/tmp/artifacts/code.py", content_type="application/python")
            }
        # Ensure global_project_settings is structured as per SharedContext for tests needing it
        # MODIFIED: update self.shared_context.data["global_project_settings"]
        self.shared_context.data["global_project_settings"] = { 
            "core_config": {"default_timeout": 300, "max_log_size": "10MB"},
            "tool_config": {"formatter": {"line_length": 88}},
            "some_global_setting": "global_value"
        }
        # Ensure initial_inputs is structured as per SharedContext
        # MODIFIED: update self.shared_context.data["initial_inputs"]
        self.shared_context.data["initial_inputs"] = {
            "user_param1": "start_value",
            "user_param2": 42
        }
        # Ensure previous_stage_outputs is structured as per SharedContext
        # MODIFIED: update self.shared_context.data["previous_stage_outputs"]
        self.shared_context.data["previous_stage_outputs"] = { 
            "prev_output_key": "value_from_previous_stage",
            "prev_list": ["a", "b", "c"]
        }

        logger.debug(f"SETUP: self.shared_context.data['outputs'] ID: {id(self.shared_context.data['outputs'])}, VALUE: {self.shared_context.data['outputs']}")

    def _add_artifact(self, artifact_id: str, content: str, metadata: Optional[Dict[str, Any]] = None, file_ext=".txt") -> Path:
        file_path = self.project_root_path / f"{artifact_id}{file_ext}"
        with open(file_path, "w") as f:
            f.write(content)
        
        path_to_store_in_artifact = str(file_path.relative_to(self.project_root_path)) # Store relative path

        artifact_detail = ArtifactDetails(
            artifact_id=artifact_id,
            name=artifact_id,
            path_on_disk=path_to_store_in_artifact, # Use the relative path string
            metadata=metadata or {},
            description=f"Test artifact {artifact_id}"
        )
        # self.shared_context.project_status.artifacts is already initialized by ProjectStateV2
        # MODIFIED: access project_status via self.shared_context.data
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts[artifact_id] = artifact_detail
        return file_path # Return absolute path for direct use in tests if needed

    def teardown_method(self, method):
        # Clean up the temporary directory
        shutil.rmtree(self.temp_dir)

    def test_resolve_literal_inputs(self) -> None:
        """Test that literal values in inputs_spec are returned as is."""
        inputs_spec = {
            "literal_str": "hello world",
            "literal_int": 123,
            "literal_bool": True,
            "literal_list": [1, 2, 3],
            "literal_dict": {"a": 1}
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved == inputs_spec, "Literal inputs should remain unchanged."

    def test_resolve_empty_inputs_spec(self) -> None:
        """Test that an empty inputs_spec results in an empty resolved dict."""
        inputs_spec: Dict[str, Any] = {}
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved == {}, "Empty inputs_spec should yield empty results."

    def test_resolve_simple_context_paths_outputs(self) -> None:
        """Test resolving simple paths from context.outputs."""
        inputs_spec = {
            "output_val": "{context.outputs.stage_one.keyA}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("output_val") == self.shared_context.data["outputs"]["stage_one"]["keyA"]

    def test_resolve_nested_context_paths_outputs(self) -> None:
        """Test resolving nested paths from context.outputs."""
        inputs_spec = {
            "nested_val": "{context.outputs.stage_one.nested_dict.deep_key}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("nested_val") == self.shared_context.data["outputs"]["stage_one"]["nested_dict"]["deep_key"]

    def test_resolve_list_index_paths_outputs(self) -> None:
        """Test resolving paths with list indices from context.outputs."""
        inputs_spec = {
            "list_item_0": "{context.outputs.stage_two.keyB[0]}",
            "list_item_1": "{context.outputs.stage_two.keyB[1]}",
            "list_item_2_sub_key": "{context.outputs.stage_two.keyB[2].sub_key}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("list_item_0") == self.shared_context.data["outputs"]["stage_two"]["keyB"][0]
        assert resolved.get("list_item_1") == self.shared_context.data["outputs"]["stage_two"]["keyB"][1]
        assert resolved.get("list_item_2_sub_key") == self.shared_context.data["outputs"]["stage_two"]["keyB"][2]["sub_key"]

    def test_resolve_mixed_literal_and_context_paths(self) -> None:
        """Test resolving a mix of literal values and context paths."""
        inputs_spec = {
            "literal_val": "iamliteral",
            "output_val": "{context.outputs.stage_one.keyA}",
            "num_literal": 42
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("literal_val") == "iamliteral"
        assert resolved.get("output_val") == self.shared_context.data["outputs"]["stage_one"]["keyA"]
        assert resolved.get("num_literal") == 42

    def test_resolve_artifact_paths(self) -> None:
        """Test resolving paths related to context.project_status.artifacts."""
        inputs_spec = {
            "artifact_path": "{context.project_status.artifacts.data_artifact.path_on_disk}",
            "artifact_type": "{context.project_status.artifacts.code_artifact.content_type}",
            "artifact_meta": "{context.project_status.artifacts.data_artifact.metadata.source}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert str(resolved.get("artifact_path")) == self.shared_context.data["project_status"].artifacts["data_artifact"].path_on_disk
        assert resolved.get("artifact_type") == self.shared_context.data["project_status"].artifacts["code_artifact"].content_type
        assert resolved.get("artifact_meta") == self.shared_context.data["project_status"].artifacts["data_artifact"].metadata["source"]

    def test_resolve_global_config_paths(self) -> None:
        """Test resolving paths from context.global_project_settings."""
        inputs_spec = {
            "core_conf_val": "{context.global_project_settings.core_config.default_timeout}",
            "tool_conf_val": "{context.global_project_settings.tool_config.formatter.line_length}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("core_conf_val") == self.shared_context.data["global_project_settings"]["core_config"]["default_timeout"]
        assert resolved.get("tool_conf_val") == self.shared_context.data["global_project_settings"]["tool_config"]["formatter"]["line_length"]

    def test_resolve_initial_inputs_paths(self) -> None:
        """Test resolving paths from context.initial_inputs."""
        inputs_spec = {
            "init_str": "{context.initial_inputs.user_param1}",
            "init_int": "{context.initial_inputs.user_param2}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("init_str") == self.shared_context.data["initial_inputs"]["user_param1"]
        assert resolved.get("init_int") == self.shared_context.data["initial_inputs"]["user_param2"]

    def test_resolve_previous_stage_outputs_paths(self) -> None:
        """Test resolving paths from context.previous_stage_outputs."""
        inputs_spec = {
            "prev_val": "{context.previous_stage_outputs.prev_output_key}",
            "prev_list_item": "{context.previous_stage_outputs.prev_list[0]}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("prev_val") == self.shared_context.data["previous_stage_outputs"]["prev_output_key"]
        assert resolved.get("prev_list_item") == self.shared_context.data["previous_stage_outputs"]["prev_list"][0]

    def test_resolve_non_existent_top_level_context_key(self) -> None:
        """Test resolving a path with a non-existent top-level key in context (e.g., context.does_not_exist)."""
        inputs_spec = {"bad_val": "{context.does_not_exist.some_key}"}
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("bad_val") is None # Based on current placeholder logic

    def test_resolve_non_existent_nested_key(self) -> None:
        """Test resolving a path with a non-existent nested key."""
        inputs_spec = {"bad_val": "{context.outputs.stage_one.non_existent_key}"}
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("bad_val") is None # Based on current placeholder logic

    def test_resolve_non_existent_list_index(self) -> None:
        """Test resolving a path with an out-of-bounds list index."""
        inputs_spec = {"bad_val": "{context.outputs.stage_two.keyB[99]}"}
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("bad_val") is None # Based on current placeholder logic

    def test_resolve_unsupported_path_format(self) -> None:
        """Test resolving a path with an unsupported or malformed format (but correctly bracketed)."""
        inputs_spec = {"bad_val": "{context.outputs.stage_one[keyA]}"} # Added closing brace
        # Depending on implementation, this might log an error and return None,
        # or return the original string, or raise an error.
        # Current placeholder returns None for unresolvable {context...} paths
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved.get("bad_val") is None
    
    # Tests for resolve_single_path method
    def test_resolve_single_path_literal_like_string(self) -> None:
        """Test resolve_single_path with a string that isn't a context path."""
        # Based on current service logic, this will warn and return a placeholder.
        # A more robust implementation might return the string itself or None.
        # For now, expecting the placeholder for non-recognized paths.
        # Actual behavior TBD by final implementation of resolve_single_path.
        # For now, this test is hard to write precisely without final logic.
        # assert self.service.resolve_single_path("not_a_context_path") == "not_a_context_path"
        pass # Revisit once resolve_single_path is more defined

    def test_resolve_single_path_simple_output(self) -> None:
        """Test resolve_single_path for a simple output."""
        val = self.service.resolve_single_path("{context.outputs.stage_one.keyA}")
        assert val == self.shared_context.data["outputs"]["stage_one"]["keyA"]

    def test_resolve_single_path_nested_output(self) -> None:
        """Test resolve_single_path for a nested output."""
        val = self.service.resolve_single_path("{context.outputs.stage_one.nested_dict.deep_key}")
        assert val == self.shared_context.data["outputs"]["stage_one"]["nested_dict"]["deep_key"]

    def test_resolve_single_path_list_index(self) -> None:
        """Test resolve_single_path for a list index."""
        assert self.service.resolve_single_path("{context.outputs.stage_two.keyB[0]}") == self.shared_context.data["outputs"]["stage_two"]["keyB"][0]
        assert self.service.resolve_single_path("{context.outputs.stage_two.keyB[1]}") == self.shared_context.data["outputs"]["stage_two"]["keyB"][1]
        # Test access within a dict that is an element of a list
        assert self.service.resolve_single_path("{context.outputs.stage_two.keyB[2].sub_key}") == self.shared_context.data["outputs"]["stage_two"]["keyB"][2]["sub_key"]

    def test_resolve_single_path_artifact_attribute(self) -> None:
        """Test resolve_single_path for an artifact attribute."""
        val = self.service.resolve_single_path("{context.project_status.artifacts.data_artifact.metadata.size_kb}")
        assert val == self.shared_context.data["project_status"].artifacts["data_artifact"].metadata["size_kb"]

    def test_resolve_single_path_global_config(self) -> None:
        """Test resolve_single_path for a global_project_settings value."""
        val = self.service.resolve_single_path("{context.global_project_settings.core_config.max_log_size}")
        assert val == self.shared_context.data["global_project_settings"]["core_config"]["max_log_size"]

    def test_resolve_single_path_initial_input(self) -> None:
        """Test resolve_single_path for an initial_input value."""
        val = self.service.resolve_single_path("{context.initial_inputs.user_param2}")
        assert val == self.shared_context.data["initial_inputs"]["user_param2"]

    def test_resolve_single_path_previous_output(self) -> None:
        """Test resolve_single_path for a previous_stage_outputs value."""
        val = self.service.resolve_single_path("{context.previous_stage_outputs.prev_list[1]}")
        assert val == self.shared_context.data["previous_stage_outputs"]["prev_list"][1]

    def test_resolve_single_path_non_existent(self) -> None:
        """Test resolve_single_path for a non-existent path."""
        val = self.service.resolve_single_path("{context.outputs.stage_one.this_key_does_not_exist}")
        assert val is None
        val_attr = self.service.resolve_single_path("{context.outputs.stage_one.keyA.non_existent_attr}")
        assert val_attr is None # Accessing attribute on a string (valueA)
        val_idx = self.service.resolve_single_path("{context.outputs.stage_two.keyB[5]}") # Index out of bounds
        assert val_idx is None
        val_key_in_list = self.service.resolve_single_path("{context.outputs.stage_two.keyB[0].some_key}") # Key access on an int
        assert val_key_in_list is None

    def test_resolve_path_with_quoted_key_access(self) -> None:
        """Test resolving paths that require quoted key access for dictionaries."""
        # Test with double quotes
        path1 = "{context.outputs.stage_with_hyphen_key[\"my-actual-key\"]}"
        resolved1 = self.service.resolve_single_path(path1)
        assert resolved1 == self.shared_context.data["outputs"]["stage_with_hyphen_key"]["my-actual-key"]

        # Test with single quotes (if regex and logic handle it)
        # For this, we might need another entry or adjust the key if SharedContext keys are simple strings
        # For now, focusing on the double-quoted one that matches the regex adjustment
        self.shared_context.data["outputs"]["stage_with_single_quote_key"] = {'another-key': "single_quote_val"}
        path2 = "{context.outputs.stage_with_single_quote_key['another-key']}"
        resolved2 = self.service.resolve_single_path(path2)
        assert resolved2 == self.shared_context.data["outputs"]["stage_with_single_quote_key"]['another-key']

        # Test in resolve_inputs_for_stage
        inputs_spec = {
            "hyphen_val_double": "{context.outputs.stage_with_hyphen_key[\"my-actual-key\"]}",
            "hyphen_val_single": "{context.outputs.stage_with_single_quote_key['another-key']}"
        }
        resolved_inputs = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved_inputs.get("hyphen_val_double") == self.shared_context.data["outputs"]["stage_with_hyphen_key"]["my-actual-key"]
        assert resolved_inputs.get("hyphen_val_single") == self.shared_context.data["outputs"]["stage_with_single_quote_key"]['another-key']

    # --- Tests for @artifact paths ---

    def test_resolve_at_path_get_artifact_object(self):
        # _add_artifact returns the Path object to the created file.
        # We need to fetch the ArtifactDetails object from shared_context.
        self._add_artifact("art1", content="content", file_ext=".txt")
        artifact_details_obj = self.shared_context.data["project_status"].artifacts.get("art1")
        
        resolved_value = self.service._resolve_at_path("@artifact.art1", self.shared_context)
        assert resolved_value == artifact_details_obj # Should return the ArtifactDetails object
        assert resolved_value.artifact_id == "art1"

    def test_resolve_at_path_artifact_path_on_disk_is_absolute_str(self):
        abs_file_path_str = str(self.project_root_path / "abs_file.txt") # ensure it's within temp dir for cleanup
        # Create the artifact first (it will have a relative path_on_disk by default)
        self._add_artifact("art_abs", content="abs content", file_ext=".txt")
        # Manually update path_on_disk to an absolute string for this test case
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts["art_abs"].path_on_disk = abs_file_path_str
        
        resolved_path = self.service._resolve_at_path("@artifact.art_abs.path", self.shared_context)
        assert isinstance(resolved_path, Path)
        assert str(resolved_path) == abs_file_path_str # Service should return Path(absolute_string)
        assert resolved_path.is_absolute()

    def test_resolve_at_path_artifact_path_on_disk_is_relative_str(self):
        # _add_artifact creates art_rel.txt and sets path_on_disk to "art_rel.txt"
        self._add_artifact("art_rel", content="rel content", file_ext=".txt")
        # Manually set a different relative path for testing this specific scenario
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts["art_rel"].path_on_disk = "relative_dir/rel_file.txt"
        
        resolved_path = self.service._resolve_at_path("@artifact.art_rel.path", self.shared_context)
        assert isinstance(resolved_path, Path)
        assert str(resolved_path) == "relative_dir/rel_file.txt" # Service should return Path(relative_string)
        assert not resolved_path.is_absolute()

    def test_resolve_at_path_artifact_path_when_path_on_disk_is_none(self):
        self._add_artifact("art_no_path", content="content for no path test")
        # Manually set path_on_disk to None
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts["art_no_path"].path_on_disk = None
            
        resolved_path = self.service._resolve_at_path("@artifact.art_no_path.path", self.shared_context)
        assert resolved_path is None

    def test_resolve_at_path_artifact_metadata_all(self):
        meta = {"key1": "val1", "key2": 123}
        self._add_artifact("art_meta", content="content for metadata test", metadata=meta)
        resolved_meta = self.service._resolve_at_path("@artifact.art_meta.metadata", self.shared_context)
        assert resolved_meta == meta

    def test_resolve_at_path_artifact_metadata_specific_key(self):
        meta = {"key1": "val1", "deep": {"nested_key": "nested_val"}}
        self._add_artifact("art_meta_key", content="content for specific key metadata", metadata=meta)
        assert self.service._resolve_at_path("@artifact.art_meta_key.metadata.key1", self.shared_context) == "val1"

    def test_resolve_at_path_artifact_metadata_nested_key(self):
        meta = {"key1": "val1", "deep": {"nested_key": "nested_val"}}
        self._add_artifact("art_meta_nested", content="content for nested key metadata", metadata=meta)
        assert self.service._resolve_at_path("@artifact.art_meta_nested.metadata.deep.nested_key", self.shared_context) == "nested_val"
        assert self.service._resolve_at_path("@artifact.art_meta_nested.metadata.deep", self.shared_context) == {"nested_key": "nested_val"}

    def test_resolve_at_path_artifact_metadata_missing_key(self):
        meta = {"key1": "val1"}
        self._add_artifact("art_meta_missing", content="content for missing key test", metadata=meta)
        with pytest.raises(KeyError): # _get_value_from_container should raise KeyError
            self.service._resolve_at_path("@artifact.art_meta_missing.metadata.non_existent_key", self.shared_context)

    def test_resolve_at_path_artifact_content_text_from_absolute_path_on_disk(self):
        # The actual file will be created by _add_artifact within project_root_path.
        # We then override path_on_disk to simulate an absolute path reference.
        created_file_path = self._add_artifact("art_content_abs", content="Hello Absolute")
        
        # For this test, make path_on_disk an absolute string pointing to the *actual* created file.
        if self.shared_context.data["project_status"]:
             self.shared_context.data["project_status"].artifacts["art_content_abs"].path_on_disk = str(created_file_path.resolve())

        # To ensure the service doesn't rely on project_root_path for absolute paths:
        original_project_root = self.shared_context.data["project_root_path"]
        self.shared_context.data["project_root_path"] = "/some/other/irrelevant/path" # Should be ignored by service for absolute path
        
        content = self.service._resolve_at_path("@artifact.art_content_abs.content", self.shared_context)
        self.shared_context.data["project_root_path"] = original_project_root # Restore
        assert content == "Hello Absolute"

    def test_resolve_at_path_artifact_content_text_from_relative_path_on_disk(self):
        # _add_artifact creates "art_content_rel.txt" inside project_root_path,
        # and sets path_on_disk to "art_content_rel.txt"
        self._add_artifact("art_content_rel", content="Hello Relative", file_ext=".txt")
        content = self.service._resolve_at_path("@artifact.art_content_rel.content", self.shared_context)
        assert content == "Hello Relative"

    def test_resolve_at_path_artifact_content_text_from_relative_path_on_disk_nested_subdir(self):
        artifact_id = "art_content_rel_nested"
        file_content = "Hello Nested Relative"
        # 1. Create artifact file at project_root
        initial_file_path = self._add_artifact(artifact_id, content=file_content, file_ext=".txt")
        
        # 2. Create subdir and move file
        subdir_path = self.project_root_path / "subdir"
        subdir_path.mkdir(exist_ok=True)
        moved_file_path = subdir_path / initial_file_path.name
        initial_file_path.rename(moved_file_path)
        
        # 3. Update artifact's path_on_disk to the new relative path
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts[artifact_id].path_on_disk = f"subdir/{initial_file_path.name}"
            
        content = self.service._resolve_at_path(f"@artifact.{artifact_id}.content", self.shared_context)
        assert content == file_content

    def test_resolve_at_path_artifact_content_file_not_found(self):
        self._add_artifact("art_no_file", content="content that won't be read")
        # Manually set path_on_disk to a non-existent file
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts["art_no_file"].path_on_disk = "non_existent_file.txt"
            
        with pytest.raises(FileNotFoundError):
            self.service._resolve_at_path("@artifact.art_no_file.content", self.shared_context)

    def test_resolve_at_path_artifact_content_path_on_disk_is_none(self):
        self._add_artifact("art_no_disk_path", content="content for no disk path test")
        if self.shared_context.data["project_status"]:
            self.shared_context.data["project_status"].artifacts["art_no_disk_path"].path_on_disk = None
            
        with pytest.raises(ValueError, match=r"Artifact \'art_no_disk_path\' path_on_disk is None, cannot read content"):
            self.service._resolve_at_path("@artifact.art_no_disk_path.content", self.shared_context)

    def test_resolve_at_path_artifact_content_no_project_root_for_relative_path(self):
        artifact_id = "art_rel_no_root"
        self._add_artifact(artifact_id, content="This should fail to resolve", file_ext=".txt")
        # path_on_disk is already relative from _add_artifact (e.g., "art_rel_no_root.txt")
        
        original_project_root = self.shared_context.data["project_root_path"]
        self.shared_context.data["project_root_path"] = None # Simulate no project root
        
        with pytest.raises(ValueError, match="project_root_path must be set in SharedContext to resolve relative artifact content path"):
            self.service._resolve_at_path(f"@artifact.{artifact_id}.content", self.shared_context)
        self.shared_context.data["project_root_path"] = original_project_root # Restore

    def test_resolve_at_path_missing_artifact(self):
        with pytest.raises(KeyError, match="Artifact 'non_existent_artifact' not found"):
            self.service._resolve_at_path("@artifact.non_existent_artifact.path", self.shared_context)

    def test_resolve_at_path_invalid_format_too_few_parts(self):
        with pytest.raises(ValueError, match=r"Invalid @artifact path format: '@artifact'. Expected @type.id\[.attribute...\]"):
            self.service._resolve_at_path("@artifact", self.shared_context)
        with pytest.raises(ValueError, match=r"Invalid @artifact path format: '@artifact.'. Item ID cannot be empty."):
            self.service._resolve_at_path("@artifact.", self.shared_context)

    def test_resolve_at_path_invalid_format_unknown_attribute(self):
        self._add_artifact("art_inv_attr", content="some content")
        with pytest.raises(ValueError, match="Unknown artifact attribute: unknown_attr"):
            self.service._resolve_at_path("@artifact.art_inv_attr.unknown_attr", self.shared_context)

    def test_resolve_inputs_for_stage_with_at_artifact_paths(self):
        artifact_id = "art1_for_inputs"
        artifact_content = "Content for art1"
        artifact_meta = {"mkey": "mval"}
        
        # _add_artifact creates file like "art1_for_inputs.txt" and sets path_on_disk relative
        created_file_path = self._add_artifact(artifact_id, content=artifact_content, metadata=artifact_meta)
        
        # Retrieve the ArtifactDetails object created by _add_artifact
        artifact_details_obj = self.shared_context.data["project_status"].artifacts[artifact_id]
        
        # For this test, we want to simulate one path_on_disk being absolute (even if it points to the same created file)
        abs_file_path_str_for_test = str(created_file_path.resolve())
        artifact_details_obj.path_on_disk = abs_file_path_str_for_test # Manually override path_on_disk to be absolute for test consistency
        
        self.shared_context.data["outputs"]["data_stage"] = {"key1": "value_from_outputs"} # Changed from context.data.key1

        inputs_spec = {
            "artifact_obj_ref": f"@artifact.{artifact_id}",
            "artifact_file_path_ref": f"@artifact.{artifact_id}.path",
            "artifact_meta_val_ref": f"@artifact.{artifact_id}.metadata.mkey",
            "artifact_data_ref": f"@artifact.{artifact_id}.content",
            "normal_data_ref": "{context.outputs.data_stage.key1}" # Changed from context.data
        }
        
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        
        assert resolved["artifact_obj_ref"] == artifact_details_obj
        assert isinstance(resolved["artifact_file_path_ref"], Path)
        assert str(resolved["artifact_file_path_ref"]) == abs_file_path_str_for_test
        assert resolved["artifact_meta_val_ref"] == "mval"
        assert resolved["artifact_data_ref"] == artifact_content
        assert resolved["normal_data_ref"] == "value_from_outputs"

    def test_get_value_from_container_dict(self):
        container = {"a": 1, "b": {"c": 2}}
        assert self.service._get_value_from_container(container, ["a"]) == 1
        assert self.service._get_value_from_container(container, ["b", "c"]) == 2
        with pytest.raises(KeyError):
            self.service._get_value_from_container(container, ["x"])
        with pytest.raises(Exception):
            self.service._get_value_from_container(container, ["a", "z"])

    def test_get_value_from_container_object(self):
        class MyObject:
            def __init__(self):
                self.a = 1
                self.b = MyNestedObject()
        class MyNestedObject:
            def __init__(self):
                self.c = 2
        
        container = MyObject()
        assert self.service._get_value_from_container(container, ["a"]) == 1
        assert self.service._get_value_from_container(container, ["b", "c"]) == 2
        with pytest.raises(AttributeError):
            self.service._get_value_from_container(container, ["x"])
        with pytest.raises(AttributeError):
            self.service._get_value_from_container(container, ["a", "z"])

    def test_get_value_from_container_list(self):
        container = [10, {"key": "val"}, [30, 40]]
        assert self.service._get_value_from_container(container, ["[0]"]) == 10
        assert self.service._get_value_from_container(container, ["[1]", "key"]) == "val"
        assert self.service._get_value_from_container(container, ["[2]", "[1]"]) == 40
        with pytest.raises(IndexError):
            self.service._get_value_from_container(container, ["[5]"])
        with pytest.raises(ValueError):
            self.service._get_value_from_container(container, ["['a']"])

    def test_resolve_single_path_complex_real_world_ish_adapted(self):
        self.shared_context.data["outputs"]["stage2_adapted"] = { # Directly assign dict, not AgentOutput model
            "output": {
                "results": [
                    {"id": "res1_adapted", "value": 100, "details": {"status": "good", "tags": ["alpha", "beta"]}},
                    {"id": "res2_adapted", "value": 200, "details": {"status": "excellent", "tags": ["gamma"]}}
                ],
                "summary_adapted": {"total_val": 300, "count": 2, "quality": "high"},
                "complex-key_adapted": {"sub-key1": [10, 20], "sub-key2": "val2_adapted"}
            }
        }
        # Test various paths
        path_expr_list_item = '{context.outputs.stage2_adapted.output.results[0].value}'
        path_expr_dict_val = '{context.outputs.stage2_adapted.output.summary_adapted.quality}'
        path_expr_complex_key = '{context.outputs.stage2_adapted.output["complex-key_adapted"]["sub-key1"][1]}'

        assert self.service.resolve_single_path(path_expr_list_item) == self.shared_context.data["outputs"]["stage2_adapted"]["output"]["results"][0]["value"]
        assert self.service.resolve_single_path(path_expr_dict_val) == self.shared_context.data["outputs"]["stage2_adapted"]["output"]["summary_adapted"]["quality"]
        assert self.service.resolve_single_path(path_expr_complex_key) == self.shared_context.data["outputs"]["stage2_adapted"]["output"]["complex-key_adapted"]["sub-key1"][1]

    def test_resolve_single_path_dict_key_with_hyphen_and_quotes_adapted(self):
        self.shared_context.data["outputs"]["stage_with_hyphen_key_adapted"] = {"my-actual-key": "hyphen_value_adapted"}
        path_expr = '{context.outputs.stage_with_hyphen_key_adapted["my-actual-key"]}'
        assert self.service.resolve_single_path(path_expr) == self.shared_context.data["outputs"]["stage_with_hyphen_key_adapted"]["my-actual-key"]

    def test_resolve_path_with_quoted_key_access_original_style(self) -> None:
        """Test resolving paths that require quoted key access for dictionaries."""
        self.shared_context.data["outputs"]["stage_with_hyphen_key"] = {"my-actual-key": "hyphen_value"}
        path1 = '{context.outputs.stage_with_hyphen_key["my-actual-key"]}'
        resolved1 = self.service.resolve_single_path(path1)
        assert resolved1 == self.shared_context.data["outputs"]["stage_with_hyphen_key"]["my-actual-key"]

    def test_resolve_single_path_dict_key_with_hyphen_and_quotes_adapted(self):
        self.shared_context.data["outputs"]["stage_with_hyphen_key_adapted"] = {"my-actual-key": "hyphen_value_adapted"}
        path_expr = '{context.outputs.stage_with_hyphen_key_adapted["my-actual-key"]}'
        assert self.service.resolve_single_path(path_expr) == self.shared_context.data["outputs"]["stage_with_hyphen_key_adapted"]["my-actual-key"]

    def test_resolve_single_path_from_global_settings(self):
        # Test resolving from global_project_settings
        path_expr = "{context.global_project_settings.some_global_setting}"
        assert self.service.resolve_single_path(path_expr) == self.shared_context.data["global_project_settings"]["some_global_setting"]

    def test_resolve_input_dict_with_literal_and_context_path(self):
        self.shared_context.data["outputs"]["some_stage"] = {"output_key": "resolved_output"}
        inputs_spec = {
            "literal_input": "literal_value",
            "path_input": "{context.outputs.some_stage.output_key}"
        }
        resolved = self.service.resolve_inputs_for_stage(inputs_spec)
        assert resolved["literal_input"] == "literal_value"
        assert resolved["path_input"] == self.shared_context.data["outputs"]["some_stage"]["output_key"]