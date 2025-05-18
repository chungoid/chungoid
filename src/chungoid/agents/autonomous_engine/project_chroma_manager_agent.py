from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal
import uuid
import collections

from pydantic import BaseModel, Field

from chungoid.utils.chroma_utils import ChromaDBManager

logger = logging.getLogger(__name__)

# --- Collection Base Names --- #
PLANNING_ARTIFACTS_COLLECTION = "planning_artifacts"
RISK_REPORTS_COLLECTION = "risk_assessment_reports"
OPTIMIZATION_REPORTS_COLLECTION = "optimization_suggestion_reports"
TRACEABILITY_REPORTS_COLLECTION = "traceability_reports"
AGENT_LOGS_COLLECTION = "agent_reflections_and_logs"
# Add more as defined in P3.M0.3.3, e.g.:
# PROJECT_GOALS_COLLECTION = "project_goals"
# LIVE_CODEBASE_COLLECTION = "live_codebase_collection"
# TEST_REPORTS_COLLECTION = "test_reports_collection"
# QA_LOGS_COLLECTION = "quality_assurance_logs"


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


# --- ProjectChromaManagerAgent Class --- #

class ProjectChromaManagerAgent_v1:
    """
    Manages project-specific ChromaDB collections for the Autonomous Project Engine.
    Provides an API for other agents to store and retrieve project artifacts.
    """
    DEFAULT_DB_SUBDIR = ".project_chroma_dbs"

    def __init__(self, project_root: Path, project_id: str, chroma_mode: str = "persistent"):
        """
        Initializes the manager for a specific project.
        Args:
            project_root: The root directory of the chungoid-core workspace.
            project_id: Unique identifier for the project.
            chroma_mode: "persistent" or "in-memory".
        """
        if not project_id:
            raise ValueError("project_id cannot be empty.")
            
        self.project_id = project_id
        self.project_root = project_root
        # Each project gets its own subdirectory within the main ChromaDB location
        self._project_db_path = project_root / self.DEFAULT_DB_SUBDIR / self.project_id
        self._project_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDBManager for this project's specific DB path
        self.chroma_manager = ChromaDBManager(
            db_directory=str(self._project_db_path),
            mode=chroma_mode
        )
        logger.info(f"ProjectChromaManagerAgent_v1 initialized for project '{project_id}' at path '{self._project_db_path}'")

    def _get_project_collection_name(self, base_collection_name: str) -> str:
        """
        Generates a project-specific collection name.
        While ChromaDBManager uses a subdirectory per project_id, 
        keeping collection names simple (without project_id prefix) is fine as they are isolated.
        This method is kept for potential future use if a single DB is used with prefixed collections.
        For now, it just returns the base_collection_name.
        """
        # return f"{self.project_id}_{base_collection_name}"
        return base_collection_name

    async def ensure_collection_exists(self, base_collection_name: str, embedding_function_name: Optional[str] = "default") -> bool:
        """
        Ensures a specific collection exists for the project.
        Args:
            base_collection_name: The base name of the collection (e.g., "planning_artifacts").
            embedding_function_name: Name of the embedding function to use if creating.
        Returns:
            True if the collection exists or was created, False on error.
        """
        collection_name = self._get_project_collection_name(base_collection_name)
        try:
            await asyncio.to_thread(self.chroma_manager.get_or_create_collection, collection_name, embedding_function_name)
            logger.debug(f"Collection '{collection_name}' ensured for project '{self.project_id}'.")
            return True
        except Exception as e:
            logger.error(f"Error ensuring collection '{collection_name}' for project '{self.project_id}': {e}", exc_info=True)
            return False

    async def store_artifact(
        self,
        args: StoreArtifactInput # Changed to use the Pydantic model
    ) -> StoreArtifactOutput:
        """
        Stores an artifact in the specified project collection.
        Handles embedding for string content if the collection is set up for it.
        Args:
            args: StoreArtifactInput containing all parameters for storing an artifact.
        Returns:
            StoreArtifactOutput indicating success or failure.
        """
        collection_name = self._get_project_collection_name(args.base_collection_name)
        doc_id = args.document_id or str(uuid.uuid4())
        
        # Make a copy of metadata to avoid modifying the input model directly
        metadata_to_store = args.metadata.copy()

        if not metadata_to_store.get("artifact_type"):
            logger.warning(f"Storing artifact in '{collection_name}' without 'artifact_type' in metadata for doc_id '{doc_id}'.")
        metadata_to_store["project_id"] = self.project_id # Ensure project_id is in metadata

        # Add lineage metadata if provided
        if args.cycle_id:
            metadata_to_store["cycle_id"] = args.cycle_id
        if args.previous_version_artifact_id:
            metadata_to_store["previous_version_artifact_id"] = args.previous_version_artifact_id
        if args.source_agent_id:
            metadata_to_store["source_agent_id"] = args.source_agent_id
        if args.source_task_id:
            metadata_to_store["source_task_id"] = args.source_task_id
        
        # Add custom linked relationships
        if args.linked_relationships:
            metadata_to_store["linked_relationships"] = args.linked_relationships

        try:
            # Ensure collection exists (using default embedding for now)
            if not await self.ensure_collection_exists(args.base_collection_name):
                 return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message=f"Failed to ensure collection '{collection_name}' exists.")

            # Prepare document content (ChromaDB expects string documents)
            if isinstance(args.artifact_content, dict):
                doc_content_str = json.dumps(args.artifact_content, indent=2)
                metadata_to_store["content_format"] = "json"
            elif isinstance(args.artifact_content, str):
                doc_content_str = args.artifact_content
                metadata_to_store["content_format"] = "text" # or "markdown" etc.
            else:
                return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message="artifact_content must be str or dict.")

            await asyncio.to_thread(
                self.chroma_manager.add_documents,
                collection_name=collection_name,
                documents=[doc_content_str],
                metadatas=[metadata_to_store], # Use the modified copy
                ids=[doc_id]
            )
            logger.info(f"Artifact '{doc_id}' (type: {metadata_to_store.get('artifact_type')}) stored in collection '{collection_name}' for project '{self.project_id}'.")
            return StoreArtifactOutput(document_id=doc_id, status="SUCCESS", message=f"Artifact '{doc_id}' stored successfully.")
        except Exception as e:
            logger.error(f"Error storing artifact '{doc_id}' in '{collection_name}': {e}", exc_info=True)
            return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message=str(e))

    async def retrieve_artifact(
        self,
        base_collection_name: str,
        document_id: str
    ) -> RetrieveArtifactOutput:
        """
        Retrieves an artifact from the specified project collection.
        Args:
            base_collection_name: Base name of the collection.
            document_id: The ID of the document to retrieve.
        Returns:
            RetrieveArtifactOutput with content, metadata, and status.
        """
        collection_name = self._get_project_collection_name(base_collection_name)
        try:
            # ChromaDBManager.get_document returns a dict with 'document', 'metadata', 'id' or None
            retrieved = await asyncio.to_thread(self.chroma_manager.get_document, collection_name, document_id)

            if retrieved and retrieved.get("document") is not None: # Check if document key exists and is not None
                content_str = retrieved["document"]
                ret_metadata = retrieved.get("metadata", {})
                
                # Attempt to parse JSON if metadata indicates it
                final_content: Union[str, Dict[str, Any]] = content_str
                if ret_metadata.get("content_format") == "json":
                    try:
                        final_content = json.loads(content_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse document '{document_id}' from '{collection_name}' as JSON, returning as string. Metadata indicated JSON.")
                
                logger.info(f"Artifact '{document_id}' retrieved from collection '{collection_name}'.")
                return RetrieveArtifactOutput(
                    document_id=document_id,
                    content=final_content,
                    metadata=ret_metadata,
                    status="SUCCESS"
                )
            else:
                logger.warning(f"Artifact '{document_id}' not found in collection '{collection_name}'.")
                return RetrieveArtifactOutput(document_id=document_id, status="NOT_FOUND", error_message="Artifact not found.")
        except Exception as e:
            logger.error(f"Error retrieving artifact '{document_id}' from '{collection_name}': {e}", exc_info=True)
            return RetrieveArtifactOutput(document_id=document_id, status="FAILURE", error_message=str(e))

    async def get_related_artifacts(self, params: GetRelatedArtifactsInput) -> GetRelatedArtifactsOutput:
        """
        Retrieves artifacts related to a concept query and/or metadata filter.
        Uses ChromaDBManager.query_collection.
        """
        collection_name = self._get_project_collection_name(params.base_collection_name)
        results_artifacts: List[RelatedArtifactItem] = [] 

        if not params.concept_query and not params.metadata_filter:
            return GetRelatedArtifactsOutput(
                status="FAILURE", 
                error_message="Either concept_query or metadata_filter must be provided."
            )

        try:
            # Ensure collection exists
            if not await self.ensure_collection_exists(params.base_collection_name):
                return GetRelatedArtifactsOutput(status="FAILURE", error_message=f"Collection '{collection_name}' does not exist or could not be created.")

            # Construct the where clause for ChromaDB query
            # If document_ids_to_exclude is provided, and metadata_filter is also provided, we need to combine them.
            # ChromaDB's where clause for excluding IDs can be tricky with $nin if the ID field isn't standard.
            # For simplicity, we'll rely on metadata_filter for direct queries and filter out excluded IDs post-query if necessary,
            # or if `concept_query` is the primary mode.
            # A more robust way would be to ensure `id` is a standard metadata field if we want to use $nin on `id`.
            # For now, if `metadata_filter` is used, it should handle ID exclusion if critical.
            # If only `concept_query` is used, we will fetch slightly more and filter.

            query_texts_list = [params.concept_query] if params.concept_query else None
            effective_n_results = params.n_results
            if params.document_ids_to_exclude and not params.metadata_filter: # If semantic search, fetch more to filter
                effective_n_results = params.n_results + len(params.document_ids_to_exclude) + 5 # Fetch a bit more

            query_results = await asyncio.to_thread(
                self.chroma_manager.query_collection,
                collection_name=collection_name,
                query_texts=query_texts_list,
                where_filter=params.metadata_filter,
                n_results=effective_n_results,
                include=["metadatas", "documents", "distances"] if params.include_content else ["metadatas", "distances"] # Include documents only if requested
            )

            if query_results:
                ids = query_results.get('ids', [[]])[0]
                docs = query_results.get('documents', [[]])[0] if params.include_content else [None] * len(ids) 
                metadatas = query_results.get('metadatas', [[]])[0]
                distances = query_results.get('distances', [[]])[0]

                for i, doc_id in enumerate(ids):
                    if doc_id in params.document_ids_to_exclude:
                        continue # Skip excluded IDs
                    
                    item_content: Optional[Union[str, Dict[str, Any]]] = None
                    if params.include_content and docs[i] is not None:
                        content_str = docs[i]
                        meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                        if meta.get("content_format") == "json":
                            try:
                                item_content = json.loads(content_str)
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON content for doc '{doc_id}' in '{collection_name}'. Returning as string.")
                                item_content = content_str
                        else:
                            item_content = content_str
                    
                    results_artifacts.append(RelatedArtifactItem(
                        document_id=doc_id,
                        content=item_content,
                        metadata=metadatas[i] if metadatas and i < len(metadatas) else None,
                        distance=distances[i] if distances and i < len(distances) else None
                    ))
                    if len(results_artifacts) >= params.n_results:
                        break # Stop once we have enough non-excluded items
            
            return GetRelatedArtifactsOutput(artifacts=results_artifacts, status="SUCCESS")

        except Exception as e:
            logger.error(f"Error querying related artifacts from '{collection_name}': {e}", exc_info=True)
            return GetRelatedArtifactsOutput(status="FAILURE", error_message=str(e))

    async def get_artifact_history(self, params: GetArtifactHistoryInput) -> GetArtifactHistoryOutput:
        """
        Retrieves the history (previous versions) of a given artifact.
        It traverses backwards using the 'previous_version_artifact_id' metadata field.
        """
        collection_name = self._get_project_collection_name(params.base_collection_name)
        versions_found: List[ArtifactVersionItem] = []
        current_doc_id: Optional[str] = params.document_id
        visited_ids = set() # To prevent infinite loops in case of bad data

        try:
            for depth in range(params.max_versions + 1): # +1 to include the starting artifact if found
                if not current_doc_id or current_doc_id in visited_ids:
                    break 
                visited_ids.add(current_doc_id)

                # We need to ensure that retrieve_artifact is called with the correct base_collection_name
                # Assuming all versions of an artifact are in the same collection as the initial artifact.
                retrieved_artifact_data = await self.retrieve_artifact(params.base_collection_name, current_doc_id)

                if retrieved_artifact_data.status == "SUCCESS":
                    # Ensure content is not None before trying to access it, even if include_content is True
                    artifact_content = None
                    if params.include_content and retrieved_artifact_data.content is not None:
                        artifact_content = retrieved_artifact_data.content
                    elif not params.include_content:
                        artifact_content = None # Explicitly None if not requested
                    
                    versions_found.append(ArtifactVersionItem(
                        document_id=retrieved_artifact_data.document_id,
                        content=artifact_content,
                        metadata=retrieved_artifact_data.metadata,
                        version_depth=depth 
                    ))
                    
                    if retrieved_artifact_data.metadata:
                        current_doc_id = retrieved_artifact_data.metadata.get("previous_version_artifact_id")
                    else:
                        current_doc_id = None 
                elif depth == 0 and retrieved_artifact_data.status == "NOT_FOUND":
                    return GetArtifactHistoryOutput(versions=[], status="NOT_FOUND", error_message=f"Starting artifact '{params.document_id}' not found in '{collection_name}'.")
                elif retrieved_artifact_data.status != "SUCCESS": # Handles FAILURE or other non-SUCCESS statuses
                    logger.warning(f"Could not retrieve version for doc_id '{current_doc_id}' (depth {depth}) in history trace for '{params.document_id}'. Status: {retrieved_artifact_data.status}, Error: {retrieved_artifact_data.error_message}")
                    break 
            
            return GetArtifactHistoryOutput(versions=versions_found, status="SUCCESS")

        except Exception as e:
            logger.error(f"Error retrieving artifact history for '{params.document_id}' from '{collection_name}': {e}", exc_info=True)
            return GetArtifactHistoryOutput(versions=[], status="FAILURE", error_message=str(e))

    async def get_artifact_genealogy(self, args: GetArtifactGenealogyInput) -> GetArtifactGenealogyOutput:
        """
        Retrieves the genealogy (lineage and relationships) of an artifact.
        Traces 'previous_version_artifact_id' and custom 'linked_relationships' in metadata.
        Args:
            args: GetArtifactGenealogyInput specifying the root artifact and traversal parameters.
        Returns:
            GetArtifactGenealogyOutput with nodes and links representing the genealogy graph.
        """
        nodes: List[ArtifactGenealogyNode] = []
        links: List[ArtifactGenealogyLink] = []
        processed_ids: set[str] = set()
        # Use collections.deque for efficient appends and pops from the left
        queue: collections.deque = collections.deque([(args.artifact_id, 0)]) 

        collection_name = self._get_project_collection_name(args.base_collection_name)

        while queue:
            current_artifact_id, current_depth = queue.popleft()

            if current_artifact_id in processed_ids:
                continue
            
            # Retrieve the artifact
            # Assuming retrieve_artifact is an async method based on others.
            # It was defined in the provided context.
            retrieved_artifact_data = await self.retrieve_artifact(
                base_collection_name=args.base_collection_name, # Use collection from input args
                document_id=current_artifact_id
            )

            if retrieved_artifact_data.status != "SUCCESS" or retrieved_artifact_data.metadata is None:
                logger.warning(f"Genealogy: Could not retrieve artifact '{current_artifact_id}' from '{collection_name}'. Status: {retrieved_artifact_data.status}")
                # Optionally, add a dummy node indicating it was not found or skip
                continue

            processed_ids.add(current_artifact_id)
            current_metadata = retrieved_artifact_data.metadata
            
            # Create and add node
            node_name = current_metadata.get("name", current_metadata.get("artifact_name", f"Artifact {current_artifact_id}"))
            if isinstance(retrieved_artifact_data.content, dict) and not current_metadata.get("name"):
                 # Try to get a name from content if it's a dict and no name in metadata
                node_name = retrieved_artifact_data.content.get("name", node_name)

            node = ArtifactGenealogyNode(
                id=current_artifact_id,
                name=str(node_name), # Ensure name is a string
                type=str(current_metadata.get("artifact_type", "UnknownType")),
                cycle_id=current_metadata.get("cycle_id"),
                source_agent_id=current_metadata.get("source_agent_id"),
                source_task_id=current_metadata.get("source_task_id"),
                raw_metadata=current_metadata
            )
            nodes.append(node)

            if current_depth < args.max_depth:
                # 1. Process 'previous_version_artifact_id'
                if args.include_previous_versions:
                    prev_version_id = current_metadata.get("previous_version_artifact_id")
                    if prev_version_id and isinstance(prev_version_id, str):
                        links.append(ArtifactGenealogyLink(
                            source_id=current_artifact_id,
                            target_id=prev_version_id,
                            relationship_type="PREVIOUS_VERSION"
                        ))
                        if prev_version_id not in processed_ids:
                            queue.append((prev_version_id, current_depth + 1))
                
                # 2. Process 'linked_relationships'
                custom_links = current_metadata.get("linked_relationships")
                if isinstance(custom_links, dict):
                    for rel_type, target_artifact_id in custom_links.items():
                        if isinstance(target_artifact_id, str): # Ensure target_id is a string
                            links.append(ArtifactGenealogyLink(
                                source_id=current_artifact_id,
                                target_id=target_artifact_id,
                                relationship_type=f"CUSTOM:{str(rel_type)}" # Ensure rel_type is string
                            ))
                            if target_artifact_id not in processed_ids:
                                queue.append((target_artifact_id, current_depth + 1))
                        else:
                            logger.warning(f"Genealogy: Invalid target_artifact_id '{target_artifact_id}' for relationship '{rel_type}' in artifact '{current_artifact_id}'. Expected string.")
        
        if not nodes and args.artifact_id not in processed_ids: # Root artifact itself was not found/processed
             return GetArtifactGenealogyOutput(
                query_artifact_id=args.artifact_id,
                status="NOT_FOUND",
                message=f"Root artifact '{args.artifact_id}' not found in collection '{collection_name}'."
            )

        return GetArtifactGenealogyOutput(
            query_artifact_id=args.artifact_id,
            nodes=nodes,
            links=links,
            status="SUCCESS",
            message=f"Genealogy retrieved with {len(nodes)} nodes and {len(links)} links."
        )

    async def initialize_project_collections(self) -> bool:
        """
        Ensures all standard collections for a project are created.
        This should be called after PCMA initialization for a new project.
        """
        standard_collections = [
            PLANNING_ARTIFACTS_COLLECTION,
            RISK_REPORTS_COLLECTION,
            OPTIMIZATION_REPORTS_COLLECTION,
            TRACEABILITY_REPORTS_COLLECTION,
            AGENT_LOGS_COLLECTION,
            # Add other standard collections here
        ]
        results = []
        for coll_name in standard_collections:
            # Using default embedding function for all initially.
            # This could be more nuanced, e.g., some collections might not need embeddings.
            success = await self.ensure_collection_exists(coll_name)
            results.append(success)
        
        if all(results):
            logger.info(f"All standard collections initialized successfully for project '{self.project_id}'.")
            return True
        else:
            logger.error(f"Failed to initialize one or more standard collections for project '{self.project_id}'.")
            return False

# Example usage (conceptual, typically other agents would use PCMA)
async def _example_pcma_usage():
    project_root_path = Path(__file__).parent.parent.parent.parent # Assuming chungoid-core
    example_project_id = "example_proj_123"
    
    pcma = ProjectChromaManagerAgent_v1(project_root=project_root_path, project_id=example_project_id)
    
    # Initialize collections (important for a new project)
    await pcma.initialize_project_collections()

    # Store an LOPRD (JSON)
    loprd_content_dict = {"version": "1.0", "goal": "Create an awesome app"}
    loprd_meta = {
        "artifact_type": "LOPRD_JSON", 
        "source_agent": "ProductAnalystAgent_v1",
        "cycle_id": "cycle_001_initial_dev",
        "source_task_id": "task_generate_loprd_abc"
    }
    store_loprd_result = await pcma.store_artifact(
        base_collection_name=PLANNING_ARTIFACTS_COLLECTION,
        artifact_content=loprd_content_dict,
        metadata=loprd_meta,
        document_id="loprd_v1"
    )
    print(f"Store LOPRD: {store_loprd_result.status}, ID: {store_loprd_result.document_id}")

    if store_loprd_result.status == "SUCCESS":
        retrieved_loprd = await pcma.retrieve_artifact(PLANNING_ARTIFACTS_COLLECTION, store_loprd_result.document_id)
        print(f"Retrieve LOPRD: {retrieved_loprd.status}, Content: {retrieved_loprd.content}")

    # Store a Blueprint (Markdown)
    blueprint_md = "# Project Blueprint\\n## Section 1..."
    blueprint_meta = {
        "artifact_type": "Blueprint_MD", 
        "source_agent": "ArchitectAgent_v1",
        "cycle_id": "cycle_001_initial_dev",
        "previous_version_artifact_id": "loprd_v1_old_version_id_example",
        "source_task_id": "task_create_blueprint_xyz"
    }
    store_bp_result = await pcma.store_artifact(
        base_collection_name=PLANNING_ARTIFACTS_COLLECTION,
        artifact_content=blueprint_md,
        metadata=blueprint_meta,
        document_id="blueprint_v1"
    )
    print(f"Store Blueprint: {store_bp_result.status}, ID: {store_bp_result.document_id}")

    if store_bp_result.status == "SUCCESS":
        retrieved_bp = await pcma.retrieve_artifact(PLANNING_ARTIFACTS_COLLECTION, store_bp_result.document_id)
        print(f"Retrieve Blueprint: {retrieved_bp.status}, Content: {retrieved_bp.content}")

if __name__ == "__main__":
    # This is for basic testing of the PCMA class structure.
    # To run this example properly, ChromaDB service needs to be accessible
    # and the project_root path must be correct.
    logging.basicConfig(level=logging.INFO)
    # Ensure the event loop is running if testing async main directly (Python 3.7+)
    # For simple script testing, you might run: asyncio.run(_example_pcma_usage())
    # However, this file is a module, so direct execution is not typical for agent operation.
    pass 