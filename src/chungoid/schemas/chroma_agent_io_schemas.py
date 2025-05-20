from __future__ import annotations
from typing import Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field

# Moved from project_chroma_manager_agent.py
class StoreArtifactOutput(BaseModel):
    document_id: str = Field(..., description="The ID of the stored/updated document in ChromaDB.")
    status: Literal["SUCCESS", "FAILURE"] = Field(..., description="Status of the store operation.")
    message: Optional[str] = None
    error_message: Optional[str] = None

class RetrieveArtifactOutput(BaseModel):
    document_id: str = Field(..., description="The ID of the retrieved document.")
    content: Optional[Union[str, Dict[str, Any]]] = Field(None, description="The content of the artifact.")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata of the artifact.")
    status: Literal["SUCCESS", "FAILURE", "NOT_FOUND"] = Field(..., description="Status of the retrieve operation.")
    error_message: Optional[str] = None

class StoreArtifactInput(BaseModel):
    base_collection_name: str = Field(..., description="Base name of the collection (e.g., PLANNING_ARTIFACTS_COLLECTION).")
    artifact_content: Union[str, Dict[str, Any]] = Field(..., description="The content (string or dict for JSON).")
    metadata: Dict[str, Any] = Field(..., description="Metadata to store with the artifact. Must include 'artifact_type'.")
    document_id: Optional[str] = Field(None, description="Optional specific ID for the artifact. If None, a UUID will be generated.")
    cycle_id: Optional[str] = Field(None, description="Optional cycle ID for lineage tracking.")
    previous_version_artifact_id: Optional[str] = Field(None, description="Optional previous version artifact ID for lineage tracking.")
    source_agent_id: Optional[str] = Field(None, description="Optional source agent ID for lineage tracking.")
    source_task_id: Optional[str] = Field(None, description="Optional source task ID for lineage tracking.")
    linked_relationships: Optional[Dict[str, str]] = Field(default_factory=dict, description="Key-value pairs representing custom relationships to other artifacts. Key is relationship type (e.g., 'DERIVED_FROM_BLUEPRINT'), value is target artifact ID.")

# NOTE: RetrieveArtifactInput is NOT moved here as it's not imported by state_manager.py.
# The test test_e2e_cli_tool_generation.py imports it from project_chroma_manager_agent.py.
# If it also causes circular dependencies later, it might need to be moved. 