from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
import uuid
import collections
import datetime

from pydantic import BaseModel, Field

# MODIFIED IMPORTS: Remove ChromaDBManager, import specific utils
from chungoid.utils.chroma_utils import (
    set_chroma_project_context,
    clear_chroma_project_context, # Good practice to have if setting context
    get_or_create_collection,
    add_or_update_document,
    get_document_by_id,
    query_documents,
    add_documents, # If batch adding is used
    # get_chroma_client # PCMA might not need to call this directly if other utils use it
)
from chungoid.utils.config_loader import get_config # For default embedding function name

from chungoid.schemas.agent_logs import ARCALogEntry

logger = logging.getLogger(__name__)

# --- Collection Base Names (as per Blueprint M0.3.3 & EXECUTION_PLAN.md Phase 3.1.3) --- #
PROJECT_GOALS_COLLECTION = "project_goals"
LOPRD_ARTIFACTS_COLLECTION = "loprd_artifacts_collection" # Used by ProductAnalystAgent
BLUEPRINT_ARTIFACTS_COLLECTION = "blueprint_artifacts_collection"
EXECUTION_PLANS_COLLECTION = "execution_plans_collection"
RISK_ASSESSMENT_REPORTS_COLLECTION = "risk_assessment_reports" # Replaces RISK_REPORTS_COLLECTION
OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION = "optimization_suggestion_reports" # Replaces OPTIMIZATION_REPORTS_COLLECTION
TRACEABILITY_REPORTS_COLLECTION = "traceability_reports" # Existing
LIVE_CODEBASE_COLLECTION = "live_codebase_collection"
GENERATED_CODE_ARTIFACTS_COLLECTION = "generated_code_artifacts"
TEST_REPORTS_COLLECTION = "test_reports_collection"
DEBUGGING_SESSION_LOGS_COLLECTION = "debugging_session_logs"
PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION = "project_documentation_artifacts"
AGENT_REFLECTIONS_AND_LOGS_COLLECTION = "agent_reflections_and_logs" # Replaces AGENT_LOGS_COLLECTION
QUALITY_ASSURANCE_LOGS_COLLECTION = "quality_assurance_logs"
LIBRARY_DOCUMENTATION_COLLECTION = "library_documentation_collection"
EXTERNAL_MCP_TOOLS_DOCUMENTATION_COLLECTION = "external_mcp_tools_documentation_collection"
REVIEW_REPORTS_COLLECTION = "review_reports_collection" # Added for blueprint reviews and similar

# List of all defined collections for easy iteration
ALL_PROJECT_COLLECTIONS = [
    PROJECT_GOALS_COLLECTION,
    LOPRD_ARTIFACTS_COLLECTION,
    BLUEPRINT_ARTIFACTS_COLLECTION,
    EXECUTION_PLANS_COLLECTION,
    RISK_ASSESSMENT_REPORTS_COLLECTION,
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
    TRACEABILITY_REPORTS_COLLECTION,
    LIVE_CODEBASE_COLLECTION,
    GENERATED_CODE_ARTIFACTS_COLLECTION,
    TEST_REPORTS_COLLECTION,
    DEBUGGING_SESSION_LOGS_COLLECTION,
    PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
    AGENT_REFLECTIONS_AND_LOGS_COLLECTION,
    QUALITY_ASSURANCE_LOGS_COLLECTION,
    LIBRARY_DOCUMENTATION_COLLECTION,
    EXTERNAL_MCP_TOOLS_DOCUMENTATION_COLLECTION,
    REVIEW_REPORTS_COLLECTION, # Added to the list
]

# --- Standardized Artifact Types (for metadata.artifact_type) --- #
# General
ARTIFACT_TYPE_GENERIC_TEXT = "GenericText"
ARTIFACT_TYPE_GENERIC_JSON = "GenericJSON"
ARTIFACT_TYPE_GENERIC_YAML = "GenericYAML"
ARTIFACT_TYPE_GENERIC_MARKDOWN = "GenericMarkdown"

# Project Lifecycle Artifacts
ARTIFACT_TYPE_PROJECT_GOAL = "ProjectGoal_MD" # For PROJECT_GOALS_COLLECTION
ARTIFACT_TYPE_LOPRD_JSON = "LOPRD_JSON" # For LOPRD_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_PROJECT_BLUEPRINT_MD = "ProjectBlueprint_MD" # For BLUEPRINT_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_MASTER_EXECUTION_PLAN_YAML = "MasterExecutionPlan_YAML" # For EXECUTION_PLANS_COLLECTION

# Reports
ARTIFACT_TYPE_RISK_ASSESSMENT_REPORT_MD = "RiskAssessmentReport_MD" # For RISK_ASSESSMENT_REPORTS_COLLECTION
ARTIFACT_TYPE_OPTIMIZATION_SUGGESTION_REPORT_MD = "OptimizationSuggestionReport_MD" # For OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION
ARTIFACT_TYPE_TRACEABILITY_MATRIX_MD = "TraceabilityMatrix_MD" # For TRACEABILITY_REPORTS_COLLECTION
ARTIFACT_TYPE_TEST_EXECUTION_REPORT_JSON = "TestExecutionReport_JSON" # For TEST_REPORTS_COLLECTION
ARTIFACT_TYPE_BLUEPRINT_REVIEW_REPORT_MD = "BlueprintReviewReport_MD" # For REVIEW_REPORTS_COLLECTION

# Code & Related
ARTIFACT_TYPE_SOURCE_CODE_FILE = "SourceCodeFile" # General, for LIVE_CODEBASE_COLLECTION (metadata can specify language)
ARTIFACT_TYPE_PYTHON_SOURCE_FILE = "PythonSourceFile"
ARTIFACT_TYPE_GENERATED_SOURCE_CODE_FILE = "GeneratedSourceCodeFile" # For GENERATED_CODE_ARTIFACTS_COLLECTION

# Documentation
ARTIFACT_TYPE_PROJECT_README_MD = "ProjectReadme_MD" # For PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_DOCS_DIRECTORY_MANIFEST_JSON = "DocsDirectoryManifest_JSON" # For PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_DEPENDENCY_AUDIT_MD = "DependencyAudit_MD" # For PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_RELEASE_NOTES_MD = "ReleaseNotes_MD" # For PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_USER_GUIDE_MD = "UserGuide_MD" # For PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
ARTIFACT_TYPE_LIBRARY_DOCS_MD = "LibraryDocs_MD" # For LIBRARY_DOCUMENTATION_COLLECTION
ARTIFACT_TYPE_EXTERNAL_TOOL_DOCS_MD = "ExternalToolDocs_MD" # For EXTERNAL_MCP_TOOLS_DOCUMENTATION_COLLECTION

# Logs & Reflections
ARTIFACT_TYPE_ARCA_LOG_ENTRY_JSON = "ARCA_LogEntry_JSON" # For AGENT_REFLECTIONS_AND_LOGS_COLLECTION
ARTIFACT_TYPE_AGENT_REFLECTION_JSON = "AgentReflection_JSON" # For AGENT_REFLECTIONS_AND_LOGS_COLLECTION
ARTIFACT_TYPE_QA_LOG_ENTRY_JSON = "QA_LogEntry_JSON" # For QUALITY_ASSURANCE_LOGS_COLLECTION
ARTIFACT_TYPE_DEBUGGING_SESSION_LOG_JSON = "DebuggingSessionLog_JSON" # For DEBUGGING_SESSION_LOGS_COLLECTION


# --- API Method Schemas (as Pydantic models) --- #

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

# --- New Schemas for get_related_artifacts --- #

class RelatedArtifactItem(BaseModel):
    document_id: str
    content: Optional[Union[str, Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None # For semantic search results

class GetRelatedArtifactsInput(BaseModel):
    base_collection_name: str = Field(..., description="Base name of the collection to query.")
    concept_query: Optional[str] = Field(None, description="Text query for semantic search to find related concepts.")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="ChromaDB 'where' clause for metadata filtering.")
    document_ids_to_exclude: Optional[List[str]] = Field(default_factory=list, description="List of document IDs to explicitly exclude from results.")
    n_results: int = Field(default=5, description="Number of results to return.")
    include_content: bool = Field(default=False, description="Whether to include the full content of the artifacts in the response. Defaults to False.")

class GetRelatedArtifactsOutput(BaseModel):
    artifacts: List[RelatedArtifactItem] = Field(default_factory=list)
    status: Literal["SUCCESS", "FAILURE"] = Field(..., description="Status of the query operation.")
    message: Optional[str] = None
    error_message: Optional[str] = None

# --- New Schemas for get_artifact_history --- #

class ArtifactVersionItem(RelatedArtifactItem):
    version_depth: int = Field(..., description="Depth of this version relative to the queried artifact. 0 is the queried artifact, 1 is its direct predecessor, etc.")

class GetArtifactHistoryInput(BaseModel):
    base_collection_name: str = Field(..., description="Base name of the collection where the artifact resides.")
    document_id: str = Field(..., description="The ID of the artifact whose history is to be retrieved.")
    max_versions: int = Field(default=5, description="Maximum number of previous versions to retrieve.")
    include_content: bool = Field(default=False, description="Whether to include the full content of the versioned artifacts.")

class GetArtifactHistoryOutput(BaseModel):
    versions: List[ArtifactVersionItem] = Field(default_factory=list)
    status: Literal["SUCCESS", "FAILURE", "NOT_FOUND"] = Field(..., description="Status of the history retrieval operation.")
    message: Optional[str] = None
    error_message: Optional[str] = None

# --- New Schemas for get_artifact_genealogy --- #

class ArtifactGenealogyNode(BaseModel):
    id: str
    name: str # Typically derived from metadata, e.g., metadata.get("name") or artifact_content if simple
    type: str # From artifact_data.metadata.artifact_type
    cycle_id: Optional[str] = None
    source_agent_id: Optional[str] = None
    source_task_id: Optional[str] = None
    raw_metadata: Dict[str, Any] = Field(default_factory=dict)

class ArtifactGenealogyLink(BaseModel):
    source_id: str
    target_id: str
    relationship_type: str # E.g., "PREVIOUS_VERSION", "CUSTOM_LINK:<original_key>"

class GetArtifactGenealogyInput(BaseModel):
    artifact_id: str = Field(..., description="The ID of the root artifact for genealogy tracing.")
    base_collection_name: str = Field(..., description="Base name of the collection where the root artifact resides.")
    max_depth: int = Field(default=2, description="Maximum depth of relationships to explore from the root artifact.")
    include_previous_versions: bool = Field(default=True, description="Whether to include 'previous_version_artifact_id' links in the genealogy.")
    # Potentially: relationship_types_to_follow: Optional[List[str]] = None

class GetArtifactGenealogyOutput(BaseModel):
    query_artifact_id: str
    nodes: List[ArtifactGenealogyNode] = Field(default_factory=list)
    links: List[ArtifactGenealogyLink] = Field(default_factory=list)
    status: Literal["SUCCESS", "FAILURE", "NOT_FOUND"] = Field(..., description="Status of the genealogy retrieval operation.")
    message: Optional[str] = None
    error_message: Optional[str] = None
    info: str = "Genealogy traces 'previous_version_artifact_id' and relationships in 'metadata.linked_relationships'."

# --- Input Schema for store_artifact --- #

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

# --- Schemas for ARCA Logging --- #

class LogStorageConfirmation(BaseModel):
    log_id: str = Field(..., description="The ID of the log entry that was processed.")
    status: Literal["SUCCESS", "FAILURE"] = Field(..., description="Status of the log storage operation.")
    message: Optional[str] = Field(None, description="Optional message providing details about the operation's outcome.")
    stored_timestamp: Optional[datetime.datetime] = Field(None, description="Timestamp when the log was actually stored, if successful.")
    error_message: Optional[str] = Field(None, description="Error message if the operation failed.")


# --- ProjectChromaManagerAgent Class --- #

class ProjectChromaManagerAgent_v1:
    """
    Manages project-specific ChromaDB collections for the Autonomous Project Engine.
    Provides an API for other agents to store and retrieve project artifacts.
    """
    DEFAULT_DB_SUBDIR = ".project_chroma_dbs" # Relative to project_root, not chungoid-core workspace root

    def __init__(self, project_root_workspace_path: Path, project_id: str):
        """
        Initializes the manager for a specific project.
        Args:
            project_root_workspace_path: The root directory of the chungoid-mcp workspace.
                                        The actual project data will be in a subdirectory.
            project_id: Unique identifier for the project (e.g., 'my_flask_app').
        """
        if not project_id:
            raise ValueError("project_id cannot be empty.")
            
        self.project_id = project_id
        # The project_root_workspace_path is the overall workspace (e.g., /home/user/chungoid-mcp)
        # The PCMA will manage its DB within a project-specific subfolder of a general chroma DB store.
        # Example: /home/user/chungoid-mcp/dev_chroma_db/my_flask_app/
        # For simplicity, we will use a fixed subdir within the workspace for all project chroma DBs.
        # This means `project_root_workspace_path` is more like the chungoid-mcp root.
        # The actual DB path for chroma_utils.set_chroma_project_context will be more specific.
        
        # Define the base directory for all project-specific ChromaDBs
        # This should align with where chroma_utils expects to find/create them based on context.
        # For persistent mode, chroma_utils uses `_current_project_directory` which is set by `set_chroma_project_context`.
        # The path given to set_chroma_project_context becomes the root for its internal .chungoid/chroma_db structure.

        # Let's assume projects are created in a standard location within the workspace, e.g., './project_workspaces/<project_id>'.
        # PCMA needs to set the context for chroma_utils to *this* specific project workspace directory.
        self.actual_project_workspace_path = project_root_workspace_path / "project_workspaces" / self.project_id
        self.actual_project_workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Set the context for chroma_utils to use this project's directory for its DB operations.
        # chroma_utils will then create its .chungoid/chroma_db within this path.
        set_chroma_project_context(self.actual_project_workspace_path)
        
        logger.info(f"ProjectChromaManagerAgent_v1 initialized for project '{project_id}'.")
        logger.info(f"ChromaDB context set to: {self.actual_project_workspace_path}")
        # No self.chroma_manager instance anymore. Calls go direct to chroma_utils functions.

    def _get_project_collection_name(self, base_collection_name: str) -> str:
        """Prepends project_id to base collection name for namespacing."""
        # return f"{self.project_id}_{base_collection_name}" # Previous method
        # Chroma_utils get_or_create_collection does not namespace by project_id automatically.
        # The namespacing is achieved by setting the DB path per project via set_chroma_project_context.
        # So, we use the base collection names directly.
        if base_collection_name not in ALL_PROJECT_COLLECTIONS:
            logger.warning(f"Base collection name '{base_collection_name}' is not in ALL_PROJECT_COLLECTIONS. Using it directly but this might be an error.")
        return base_collection_name

    async def ensure_collection_exists(self, base_collection_name: str, embedding_function_name: Optional[str] = None) -> bool:
        """
        Ensures a collection exists for the project. The actual collection in ChromaDB
        will be project-specific due to the DB path context.
        Args:
            base_collection_name: The base name for the collection.
            embedding_function_name: Optional name of the embedding function.
        Returns:
            True if collection exists or was created, False otherwise.
        """
        # Set context just in case it was cleared or changed by another PCMA instance for a different project.
        # This makes each PCMA call relatively stateless regarding context setting for chroma_utils.
        set_chroma_project_context(self.actual_project_workspace_path)
        
        collection_name_to_ensure = self._get_project_collection_name(base_collection_name)
        try:
            # embedding_function_name is not directly used by get_or_create_collection in chroma_utils
            # but chroma_utils.get_chroma_client() might pick up default embedding from config.
            collection = get_or_create_collection(collection_name=collection_name_to_ensure)
            return collection is not None
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name_to_ensure}' for project '{self.project_id}': {e}", exc_info=True)
            return False
    
    async def store_artifact(
        self,
        args: StoreArtifactInput # Changed to use the Pydantic model
    ) -> StoreArtifactOutput:
        """
        Stores or updates an artifact in the project-specific collection.
        The content is converted to JSON string if it's a dict.
        Manages lineage and other common metadata fields.
        """
        set_chroma_project_context(self.actual_project_workspace_path)
        collection_name = self._get_project_collection_name(args.base_collection_name)
        
        doc_id = args.document_id or str(uuid.uuid4())
        
        content_to_store = (
            json.dumps(args.artifact_content, indent=2)
            if isinstance(args.artifact_content, dict)
            else str(args.artifact_content) # Ensure it's a string
        )

        final_metadata = args.metadata.copy()
        final_metadata["project_id"] = self.project_id
        final_metadata["artifact_id"] = doc_id # Store the doc_id in metadata as well for easier querying
        final_metadata["last_modified_utc"] = datetime.datetime.utcnow().isoformat()
        if args.cycle_id:
            final_metadata["cycle_id"] = args.cycle_id
        if args.source_agent_id:
            final_metadata["source_agent_id"] = args.source_agent_id
        if args.source_task_id:
            final_metadata["source_task_id"] = args.source_task_id
        if args.previous_version_artifact_id:
            final_metadata["previous_version_artifact_id"] = args.previous_version_artifact_id
        if args.linked_relationships:
            final_metadata["linked_relationships"] = json.dumps(args.linked_relationships) # Store as JSON string if complex

        if "artifact_type" not in final_metadata:
            logger.warning(f"Storing artifact '{doc_id}' in '{collection_name}' without 'artifact_type' in metadata.")
            # return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message="'artifact_type' is required in metadata.")

        try:
            # Use add_or_update_document from chroma_utils
            success = add_or_update_document(
                collection_name=collection_name,
                doc_id=doc_id,
                document_content=content_to_store,
                metadata=final_metadata,
            )
            if success:
                return StoreArtifactOutput(document_id=doc_id, status="SUCCESS", message=f"Artifact '{doc_id}' stored in '{collection_name}'.")
            else:
                # This path might not be hit if add_or_update_document raises exceptions on failure
                return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message=f"Failed to store artifact '{doc_id}' in '{collection_name}'.")
        except Exception as e:
            logger.error(f"Error storing artifact '{doc_id}' in '{collection_name}' for project '{self.project_id}': {e}", exc_info=True)
            return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message=str(e))

    async def retrieve_artifact(
        self,
        base_collection_name: str,
        document_id: str
    ) -> RetrieveArtifactOutput:
        """Retrieves an artifact by its ID from the project-specific collection."""
        set_chroma_project_context(self.actual_project_workspace_path)
        collection_name = self._get_project_collection_name(base_collection_name)
        try:
            # Use get_document_by_id from chroma_utils
            result = get_document_by_id(
                collection_name=collection_name, 
                doc_id=document_id,
                include=["documents", "metadatas"] # Ensure both are included
            )
            if result and result.get("ids") and result.get("documents"):
                retrieved_doc_id = result["ids"][0]
                content_str = result["documents"][0]
                metadata = result["metadatas"][0] if result.get("metadatas") else {}
                
                # Try to parse content as JSON if it looks like it
                parsed_content: Union[str, Dict[str, Any]] = content_str
                try:
                    if content_str.strip().startswith(("{", "[")):
                        parsed_content = json.loads(content_str)
                except json.JSONDecodeError:
                    pass # Keep as string if not valid JSON

                return RetrieveArtifactOutput(
                    document_id=retrieved_doc_id,
                    content=parsed_content,
                    metadata=metadata,
                    status="SUCCESS"
                )
            else:
                logger.warning(f"Artifact '{document_id}' not found in collection '{collection_name}' for project '{self.project_id}'. Result: {result}")
                return RetrieveArtifactOutput(document_id=document_id, status="NOT_FOUND", error_message="Artifact not found.")
        except Exception as e:
            logger.error(f"Error retrieving artifact '{document_id}' from '{collection_name}' for project '{self.project_id}': {e}", exc_info=True)
            return RetrieveArtifactOutput(document_id=document_id, status="FAILURE", error_message=str(e))

    async def get_related_artifacts(self, params: GetRelatedArtifactsInput) -> GetRelatedArtifactsOutput:
        """
        Queries a collection for artifacts related to a concept or matching metadata filters.
        Excludes specified document IDs from the results.
        """
        set_chroma_project_context(self.actual_project_workspace_path)
        collection_name = self._get_project_collection_name(params.base_collection_name)
        related_items: List[RelatedArtifactItem] = []
        
        # Adjust n_results if we need to filter out excluded IDs later
        # This is a simple approach; more sophisticated handling might query more and then filter.
        query_n_results = params.n_results + len(params.document_ids_to_exclude or [])
        if query_n_results > 20: # Cap to avoid excessive queries
            query_n_results = 20 

        try:
            # Use query_documents from chroma_utils
            query_results = query_documents(
                collection_name=collection_name,
                query_texts=[params.concept_query] if params.concept_query else None,
                n_results=query_n_results, # Fetch a bit more to allow for filtering
                where_filter=params.metadata_filter,
                include=["documents", "metadatas", "distances"]
            )

            if query_results and query_results.get("ids") and query_results.get("documents"):
                num_results = len(query_results["ids"][0])
                for i in range(num_results):
                    doc_id = query_results["ids"][0][i]
                    if doc_id in (params.document_ids_to_exclude or []):
                        continue # Skip excluded IDs

                    if len(related_items) >= params.n_results:
                        break # Reached desired number of results after filtering

                    content_str = query_results["documents"][0][i] if query_results.get("documents") and query_results["documents"][0] else None
                    metadata = query_results["metadatas"][0][i] if query_results.get("metadatas") and query_results["metadatas"][0] else {}
                    distance = query_results["distances"][0][i] if query_results.get("distances") and query_results["distances"][0] else None

                    parsed_content: Optional[Union[str, Dict[str, Any]]] = None
                    if params.include_content and content_str:
                        parsed_content = content_str
                        try:
                            if content_str.strip().startswith(("{", "[")):
                                parsed_content = json.loads(content_str)
                        except json.JSONDecodeError:
                            pass # Keep as string
                    
                    related_items.append(RelatedArtifactItem(
                        document_id=doc_id,
                        content=parsed_content if params.include_content else None,
                        metadata=metadata,
                        distance=distance
                    ))
            
            return GetRelatedArtifactsOutput(artifacts=related_items, status="SUCCESS", message=f"Found {len(related_items)} related artifacts.")

        except Exception as e:
            logger.error(f"Error querying related artifacts in '{collection_name}': {e}", exc_info=True)
            return GetRelatedArtifactsOutput(artifacts=[], status="FAILURE", error_message=str(e))

    # ... (get_artifact_history and get_artifact_genealogy would also need similar refactoring
    #      to use chroma_utils.get_document_by_id and potentially chroma_utils.query_documents
    #      instead of self.chroma_manager. ...)
    # Placeholder for brevity, these need full implementation using chroma_utils
    async def get_artifact_history(self, params: GetArtifactHistoryInput) -> GetArtifactHistoryOutput:
        set_chroma_project_context(self.actual_project_workspace_path)
        logger.warning("get_artifact_history is not fully implemented with new chroma_utils.")
        # Basic logic: Start with the given doc_id, then iteratively call get_document_by_id 
        # for 'previous_version_artifact_id' found in metadata.
        versions: List[ArtifactVersionItem] = []
        current_doc_id = params.document_id
        processed_ids = set()

        for i in range(params.max_versions + 1): # +1 to fetch the initial doc
            if not current_doc_id or current_doc_id in processed_ids:
                break
            processed_ids.add(current_doc_id)

            try:
                retrieved_doc = await self.retrieve_artifact(params.base_collection_name, current_doc_id)
                if retrieved_doc.status == "SUCCESS" and retrieved_doc.document_id:
                    content_to_include = retrieved_doc.content if params.include_content else None
                    versions.append(ArtifactVersionItem(
                        document_id=retrieved_doc.document_id,
                        content=content_to_include,
                        metadata=retrieved_doc.metadata,
                        version_depth=i
                    ))
                    current_doc_id = retrieved_doc.metadata.get("previous_version_artifact_id") if retrieved_doc.metadata else None
                else:
                    if i == 0: # If the initial document is not found
                         return GetArtifactHistoryOutput(versions=[], status="NOT_FOUND", message=f"Initial artifact {params.document_id} not found.")
                    break # Stop if a previous version is not found
            except Exception as e:
                logger.error(f"Error retrieving version {i} (doc_id: {current_doc_id}) for history of {params.document_id}: {e}", exc_info=True)
                return GetArtifactHistoryOutput(versions=versions, status="FAILURE", error_message=str(e))
        
        return GetArtifactHistoryOutput(versions=versions, status="SUCCESS")

    async def get_artifact_genealogy(self, args: GetArtifactGenealogyInput) -> GetArtifactGenealogyOutput:
        set_chroma_project_context(self.actual_project_workspace_path)
        logger.warning("get_artifact_genealogy is not fully implemented with new chroma_utils.")
        # This method is complex and would require careful traversal of linked_relationships and previous_version_artifact_id
        # using multiple calls to self.retrieve_artifact (which uses chroma_utils.get_document_by_id)
        # For now, returning a placeholder.
        nodes: List[ArtifactGenealogyNode] = []
        links: List[ArtifactGenealogyLink] = []
        q = collections.deque([(args.artifact_id, 0)]) # (doc_id, depth)
        visited_ids = set()

        initial_artifact_retrieved = False

        while q:
            current_doc_id, depth = q.popleft()

            if current_doc_id in visited_ids or depth > args.max_depth:
                continue
            visited_ids.add(current_doc_id)

            try:
                # Use self.retrieve_artifact which is already refactored
                artifact_data = await self.retrieve_artifact(args.base_collection_name, current_doc_id)
                
                if artifact_data.status != "SUCCESS" or not artifact_data.metadata or not artifact_data.document_id:
                    if not initial_artifact_retrieved and current_doc_id == args.artifact_id:
                        return GetArtifactGenealogyOutput(query_artifact_id=args.artifact_id, status="NOT_FOUND", message=f"Root artifact {args.artifact_id} not found.")
                    logger.warning(f"Could not retrieve artifact {current_doc_id} for genealogy.")
                    continue
                
                if not initial_artifact_retrieved and current_doc_id == args.artifact_id:
                    initial_artifact_retrieved = True

                node_name = artifact_data.metadata.get("name", artifact_data.metadata.get("artifact_type", current_doc_id))
                if isinstance(artifact_data.content, str) and len(artifact_data.content) < 50 and not node_name:
                    node_name = artifact_data.content # Fallback name for short content
                
                nodes.append(ArtifactGenealogyNode(
                    id=artifact_data.document_id,
                    name=node_name,
                    type=artifact_data.metadata.get("artifact_type", "Unknown"),
                    cycle_id=artifact_data.metadata.get("cycle_id"),
                    source_agent_id=artifact_data.metadata.get("source_agent_id"),
                    source_task_id=artifact_data.metadata.get("source_task_id"),
                    raw_metadata=artifact_data.metadata
                ))

                # Process previous version link
                if args.include_previous_versions:
                    prev_version_id = artifact_data.metadata.get("previous_version_artifact_id")
                    if prev_version_id and prev_version_id not in visited_ids: # Check visited_ids here too
                        links.append(ArtifactGenealogyLink(source_id=prev_version_id, target_id=current_doc_id, relationship_type="PREVIOUS_VERSION"))
                        if depth + 1 <= args.max_depth:
                            q.append((prev_version_id, depth + 1))
                
                # Process custom linked relationships
                linked_rels_str = artifact_data.metadata.get("linked_relationships")
                if linked_rels_str:
                    try:
                        linked_rels: Dict[str, str] = json.loads(linked_rels_str) if isinstance(linked_rels_str, str) else linked_rels_str
                        for rel_type, target_id in linked_rels.items():
                            if target_id and target_id not in visited_ids: # Check visited_ids here too
                                links.append(ArtifactGenealogyLink(source_id=target_id, target_id=current_doc_id, relationship_type=f"CUSTOM_LINK:{rel_type}"))
                                # Also add reverse link for visualization or allow user to specify directionality?
                                # For now, assume links point *to* current_doc_id from the target_id in metadata key
                                # If we want to explore *from* current_doc_id *to* its links, we need to store them differently or query differently.
                                # The current design means `linked_relationships` are "things this artifact is linked FROM"
                                # To trace outwards, the source artifact would need to list its targets.
                                # For now, we are tracing backwards based on these links.
                                if depth + 1 <= args.max_depth:
                                     q.append((target_id, depth + 1))
                    except json.JSONDecodeError:
                        logger.warning(f"Could not parse linked_relationships for {current_doc_id}: {linked_rels_str}")

            except Exception as e:
                logger.error(f"Error processing artifact {current_doc_id} for genealogy: {e}", exc_info=True)
                # Continue processing other branches if possible
        
        if not initial_artifact_retrieved and args.artifact_id and not any(n.id == args.artifact_id for n in nodes):
             return GetArtifactGenealogyOutput(query_artifact_id=args.artifact_id, status="NOT_FOUND", message=f"Root artifact {args.artifact_id} could not be processed or found.")

        return GetArtifactGenealogyOutput(query_artifact_id=args.artifact_id, nodes=nodes, links=links, status="SUCCESS")


    async def initialize_project_collections(self) -> bool:
        """
        Ensures all standard project collections are initialized in ChromaDB for the current project.
        This should be called when a new project is created or to verify existing setup.
        Uses the default embedding function specified in the global config or ChromaDB's default.
        """
        set_chroma_project_context(self.actual_project_workspace_path)
        logger.info(f"Initializing all standard collections for project: {self.project_id}")
        
        app_config = get_config()
        chroma_db_config = app_config.get("chromadb", {})
        default_embedding_function_name = chroma_db_config.get("default_embedding_function")
        
        all_successful = True
        for base_collection_name in ALL_PROJECT_COLLECTIONS:
            project_collection_name = self._get_project_collection_name(base_collection_name)
            try:
                logger.debug(f"Ensuring collection: {project_collection_name} with embedding: {default_embedding_function_name or 'ChromaDefault'}")
                # ensure_collection_exists uses get_or_create_collection from chroma_utils
                success = await self.ensure_collection_exists(
                    base_collection_name=base_collection_name,
                    embedding_function_name=default_embedding_function_name
                )
                if not success:
                    all_successful = False
                    logger.error(f"Failed to ensure existence of collection: {project_collection_name}")
            except Exception as e:
                all_successful = False
                logger.error(f"Exception while ensuring collection {project_collection_name}: {e}", exc_info=True)
        
        if all_successful:
            logger.info(f"Successfully initialized/verified all standard collections for project {self.project_id}.")
        else:
            logger.warning(f"One or more collections could not be initialized/verified for project {self.project_id}.")
        return all_successful

    async def log_arca_event(
        self,
        project_id: str, # For validation against self.project_id
        cycle_id: str,   # For metadata or partitioning, if PCMA uses it
        log_entry: ARCALogEntry
    ) -> LogStorageConfirmation:
        """
        Stores an ARCALogEntry into the AGENT_REFLECTIONS_AND_LOGS_COLLECTION.
        Args:
            project_id: The ID of the project this log belongs to.
            cycle_id: The specific autonomous cycle ID this log is associated with.
            log_entry: The ARCALogEntry Pydantic model instance.
        Returns:
            LogStorageConfirmation indicating success or failure.
        """
        set_chroma_project_context(self.actual_project_workspace_path)
        if project_id != self.project_id:
            err_msg = f"Mismatched project_id in log_arca_event. Expected '{self.project_id}', got '{project_id}'."
            logger.error(err_msg)
            return LogStorageConfirmation(log_id=log_entry.log_id, status="FAILURE", error_message=err_msg)

        collection_name = self._get_project_collection_name(AGENT_REFLECTIONS_AND_LOGS_COLLECTION)
        
        document_content = log_entry.model_dump_json(indent=2)
        doc_id = f"arca_log_{log_entry.log_id}"
        
        metadata = {
            "artifact_type": "ARCA_LogEntry",
            "project_id": self.project_id,
            "cycle_id": cycle_id,
            "agent_id": log_entry.agent_id,
            "task_id": log_entry.task_id,
            "event_type": log_entry.event_type.value, # Store enum value
            "timestamp_utc": log_entry.timestamp_utc.isoformat(),
            "log_id_prop": log_entry.log_id # Adding log_id also as a direct metadata field for easier query
        }
        if log_entry.related_artifact_id:
            metadata["related_artifact_id"] = log_entry.related_artifact_id

        try:
            success = add_or_update_document(
                collection_name=collection_name,
                doc_id=doc_id,
                document_content=document_content,
                metadata=metadata,
            )
            if success:
                msg = f"ARCA LogEntry '{log_entry.log_id}' stored successfully in '{collection_name}'."
                logger.debug(msg)
                return LogStorageConfirmation(
                    log_id=log_entry.log_id, 
                    status="SUCCESS", 
                    message=msg,
                    stored_timestamp=datetime.datetime.utcnow() # Actual store time
                )
            else:
                err_msg = f"Failed to store ARCA LogEntry '{log_entry.log_id}' in '{collection_name}' (add_or_update_document returned False)."
                logger.error(err_msg)
                return LogStorageConfirmation(log_id=log_entry.log_id, status="FAILURE", error_message=err_msg)
        except Exception as e:
            logger.error(f"Error storing ARCA LogEntry '{log_entry.log_id}' in '{collection_name}': {e}", exc_info=True)
            return LogStorageConfirmation(log_id=log_entry.log_id, status="FAILURE", error_message=str(e))

    # Optional: Add a destructor or cleanup method if needed, e.g., to clear context for this project.
    # def __del__(self):
    #     # This might be called when the PCMA instance is garbage collected.
    #     # Be cautious with global context manipulation here if multiple PCMA instances for different projects
    #     # might exist and be GC'd unpredictably.
    #     # For now, explicit context setting per call is safer.
    #     # if _current_project_directory == self.actual_project_workspace_path:
    #     #     clear_chroma_project_context() # Clear context if it was for this project
    #     pass


# Example Usage (Conceptual - for understanding, not direct execution here)
async def _example_pcma_usage():
    # This would typically be in a higher-level orchestrator or test script
    workspace_root = Path(__file__).resolve().parents[3] # Adjust to your actual workspace root
    project_name = "example_project_for_pcma"

    # Create a PCMA instance for a specific project
    pcma = ProjectChromaManagerAgent_v1(project_root_workspace_path=workspace_root, project_id=project_name)

    # Initialize collections for the project
    await pcma.initialize_project_collections()

    # Store an artifact
    loprd_content = {"goal": "Test LOPRD for PCMA", "requirements": ["Req1"]}
    store_input = StoreArtifactInput(
        base_collection_name=LOPRD_ARTIFACTS_COLLECTION,
        artifact_content=loprd_content,
        metadata={"artifact_type": "LOPRD_JSON", "author": "Test Script"},
        cycle_id="cycle_001",
        source_agent_id="TestAgent_v1"
    )
    store_result = await pcma.store_artifact(store_input)
    print(f"Store result: {store_result.model_dump_json(indent=2)}")

    if store_result.status == "SUCCESS":
        doc_id = store_result.document_id
        # Retrieve the artifact
        retrieve_result = await pcma.retrieve_artifact(LOPRD_ARTIFACTS_COLLECTION, doc_id)
        print(f"Retrieve result: {retrieve_result.model_dump_json(indent=2)}")

        if retrieve_result.status == "SUCCESS" and isinstance(retrieve_result.content, dict):
            assert retrieve_result.content["goal"] == "Test LOPRD for PCMA"

    # Example of ARCA logging
    from chungoid.schemas.agent_logs import ARCAEventType # Assuming enum definition
    log_event = ARCALogEntry(
        log_id=str(uuid.uuid4()),
        agent_id="ARCA_v1",
        task_id="task_abc",
        event_type=ARCAEventType.ARCA_DECISION_MADE,
        summary="LOPRD approved, proceeding to blueprinting.",
        details={"loprd_doc_id": store_result.document_id if store_result.status == "SUCCESS" else "N/A", "confidence": 0.95}
    )
    log_conf = await pcma.log_arca_event(project_id=project_name, cycle_id="cycle_001", log_entry=log_event)
    print(f"Log confirmation: {log_conf.model_dump_json(indent=2)}")

# if __name__ == "__main__":
#     asyncio.run(_example_pcma_usage()) 