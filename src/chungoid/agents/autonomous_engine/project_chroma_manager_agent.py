from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import uuid

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
        base_collection_name: str,
        artifact_content: Union[str, Dict[str, Any]],
        metadata: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> StoreArtifactOutput:
        """
        Stores an artifact in the specified project collection.
        Handles embedding for string content if the collection is set up for it.
        Args:
            base_collection_name: Base name of the collection (e.g., PLANNING_ARTIFACTS_COLLECTION).
            artifact_content: The content (string or dict for JSON).
            metadata: Metadata to store with the artifact. Must include 'artifact_type'.
            document_id: Optional specific ID for the artifact. If None, a UUID will be generated.
        Returns:
            StoreArtifactOutput indicating success or failure.
        """
        collection_name = self._get_project_collection_name(base_collection_name)
        doc_id = document_id or str(uuid.uuid4())
        
        if not metadata.get("artifact_type"):
            logger.warning(f"Storing artifact in '{collection_name}' without 'artifact_type' in metadata for doc_id '{doc_id}'.")
        metadata["project_id"] = self.project_id # Ensure project_id is in metadata

        try:
            # Ensure collection exists (using default embedding for now)
            if not await self.ensure_collection_exists(base_collection_name):
                 return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message=f"Failed to ensure collection '{collection_name}' exists.")

            # Prepare document content (ChromaDB expects string documents)
            if isinstance(artifact_content, dict):
                doc_content_str = json.dumps(artifact_content, indent=2)
                metadata["content_format"] = "json"
            elif isinstance(artifact_content, str):
                doc_content_str = artifact_content
                metadata["content_format"] = "text" # or "markdown" etc.
            else:
                return StoreArtifactOutput(document_id=doc_id, status="FAILURE", error_message="artifact_content must be str or dict.")

            await asyncio.to_thread(
                self.chroma_manager.add_documents,
                collection_name=collection_name,
                documents=[doc_content_str],
                metadatas=[metadata],
                ids=[doc_id]
            )
            logger.info(f"Artifact '{doc_id}' (type: {metadata.get('artifact_type')}) stored in collection '{collection_name}' for project '{self.project_id}'.")
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

    # --- Placeholder for other methods from design doc ---
    # async def update_artifact_metadata(...) -> StoreArtifactOutput: raise NotImplementedError
    # async def query_collection_by_metadata(...) -> QueryCollectionOutput: raise NotImplementedError
    # async def query_collection_by_text(...) -> QueryCollectionOutput: raise NotImplementedError
    # async def delete_artifact(...) -> StoreArtifactOutput: raise NotImplementedError

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
    loprd_meta = {"artifact_type": "LOPRD_JSON", "source_agent": "ProductAnalystAgent_v1"}
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
    blueprint_meta = {"artifact_type": "Blueprint_MD", "source_agent": "ArchitectAgent_v1"}
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