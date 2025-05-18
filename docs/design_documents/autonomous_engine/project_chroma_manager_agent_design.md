---
title: "Agent Design: ProjectChromaManagerAgent_v1"
category: design_document
owner: "Meta Engineering Team"
status: "Draft"
created: YYYY-MM-DD # To be filled
updated: YYYY-MM-DD # To be filled
version: "0.1.0"
related_blueprint: "blueprint_autonomous_project_engine.md"
related_agents: ["ProductAnalystAgent_v1", "ArchitectAgent", "BlueprintReviewerAgent", "BlueprintToFlowAgent", "CoreCodeGeneratorAgent_v1", "CoreTestGeneratorAgent_v1", "CoreProjectDocumentationAgent", "ProactiveRiskAssessorAgent", "AutomatedRefinementCoordinatorAgent", "RequirementsTracerAgent"]
---

# Agent Design: ProjectChromaManagerAgent_v1

## 0. Document History
| Version | Date       | Author             | Changes                                      |
|---------|------------|--------------------|----------------------------------------------|
| 0.1.0   | YYYY-MM-DD | Gemini Assistant   | Initial Draft                                |

## 1. Purpose & Scope

### 1.1. Purpose
The `ProjectChromaManagerAgent_v1` (PCMA) acts as the centralized interface for all interactions with project-specific ChromaDB collections within the Autonomous Project Engine. It encapsulates the logic for creating, retrieving, updating, and managing various project artifacts and contextual data stored in vector databases. This ensures consistency, manageability, and a clear separation of concerns for data persistence operations across all other agents in the engine.

### 1.2. Scope
#### 1.2.1. In Scope
*   Managing (CRUD operations, querying) dedicated ChromaDB collections for a specific project lifecycle.
*   Storing and retrieving key project artifacts such as:
    *   Refined User Goals (`refined_user_goal.md`, `assumptions_and_ambiguities.md`)
    *   LOPRDs (`llm_optimized_prd.json`)
    *   Project Blueprints (`ProjectBlueprint.md`)
    *   Master Execution Plans (`MasterExecutionPlan.yaml`)
    *   Schemas (e.g., `loprd_schema.json` if project-specific or versioned)
    *   Generated Code (source files, embeddings)
    *   Test Reports
    *   Documentation (project-level, code-level comments/embeddings)
    *   Risk Assessment Reports
    *   Traceability Reports
    *   Optimization Suggestion Reports
    *   Agent Reflections and Logs (operational logs, decision rationale, confidence scores)
    *   Quality Assurance Logs
    *   External Library Documentation (embeddings)
    *   External MCP Tool Documentation (embeddings)
*   Providing an API (likely Python methods) for other agents to interact with these collections without needing direct ChromaDB client knowledge.
*   Handling document embedding (if not done by the calling agent) before storage for relevant artifact types.
*   Ensuring consistent metadata is stored alongside artifacts (e.g., artifact ID, version, timestamp, source agent, project ID).
*   Initializing project-specific ChromaDB collections if they don't exist.

#### 1.2.2. Out of Scope
*   Acting as a general-purpose ChromaDB administration tool for the entire Chungoid system (its focus is project-specific within the Autonomous Project Engine).
*   Defining the schemas of the artifacts it stores (it consumes schemas defined elsewhere, e.g., `loprd_schema.json`). It may, however, manage a collection of schema definitions if needed.
*   Complex data analysis or inference on the stored data (other agents perform analysis).
*   Managing the lifecycle of the ChromaDB service itself (e.g., starting/stopping the server).
*   Authentication and authorization to ChromaDB (assumed to be handled at a lower level or by the environment configuration).
*   Direct interaction with LLMs (other agents handle LLM interactions).

## 2. High-Level Architecture

```mermaid
graph TD
    subgraph ProjectChromaManagerAgent_v1 as PCMA_Subgraph
        direction LR
        PCMA_API[API Interface (Python Methods)]
        PCMA_Logic[Core Logic (Collection Mgmt, CRUD, Querying)]
        ChromaClient[ChromaDB Client]
    end

    OtherAgent[Other Autonomous Engine Agents] -- API Calls --> PCMA_API
    PCMA_API --> PCMA_Logic
    PCMA_Logic --> ChromaClient
    ChromaClient <--> ChromaDB[(ChromaDB Instance)]

    ChromaDB -- Stores --> Col1[Collection: project_goals]
    ChromaDB -- Stores --> Col2[Collection: project_planning_artifacts]
    ChromaDB -- Stores --> Col3[Collection: live_codebase_collection]
    ChromaDB -- Stores --> Col4[Collection: agent_reflections_and_logs]
    ChromaDB -- Stores --> ColN[Collection: ...etc.]
```
*The `ProjectChromaManagerAgent_v1` exposes a Python API to other agents. Its internal logic uses a ChromaDB client to interact with various project-specific collections within a ChromaDB instance.*

## 3. Agent Responsibilities & Capabilities

### 3.1. Core Responsibilities
*   Provide a stable and consistent interface for project data persistence.
*   Abstract away the direct complexities of ChromaDB interaction for other agents.
*   Ensure artifacts are stored with appropriate metadata for traceability and context.
*   Manage the set of predefined ChromaDB collections required for a project generated by the Autonomous Project Engine.

### 3.2. Key Capabilities
*   **Artifact Storage:** Store various types of documents and data (text, JSON, code) into specified ChromaDB collections, potentially handling embedding generation.
*   **Artifact Retrieval:** Retrieve artifacts by ID, or query collections based on metadata or semantic similarity (vector search).
*   **Artifact Update:** Support updating existing artifacts (e.g., new versions of LOPRDs, code files).
*   **Artifact Deletion:** Support removal of artifacts if necessary.
*   **Collection Management:** Ensure necessary collections exist for a given project context.
*   **Metadata Association:** Store and retrieve rich metadata associated with each artifact.

## 4. Input/Output Schemas (Illustrative for API Methods)

*(This agent is not typically orchestrated in the same way as LLM-based agents. It provides utility functions/methods. The schemas below are illustrative of the Pydantic models that might define the inputs and outputs of its API methods.)*

### 4.1. Example API Method: `store_artifact`
```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Union, Optional

class StoreArtifactInput(BaseModel):
    project_id: str = Field(..., description="Identifier for the current project.")
    collection_name: str = Field(..., description="Target ChromaDB collection name.")
    artifact_id: Optional[str] = Field(None, description="Specific ID for the artifact; if None, one might be generated.")
    artifact_content: Union[str, Dict[str, Any]] = Field(..., description="The content of the artifact (text, JSON object). Embedding happens internally if needed.")
    metadata: Dict[str, Any] = Field(..., description="Metadata to store with the artifact (e.g., source_agent_id, version, timestamp, artifact_type).")
    # add_to_embedding_queue: bool = Field(True, description="Whether to embed this artifact if it's text-based.")

class StoreArtifactOutput(BaseModel):
    document_id: str # The ID in ChromaDB (could be the same as artifact_id or generated)
    status: str # SUCCESS/FAILURE
    error_message: Optional[str] = None
```

### 4.2. Example API Method: `retrieve_artifact`
```python
class RetrieveArtifactInput(BaseModel):
    project_id: str
    collection_name: str
    document_id: str

class RetrieveArtifactOutput(BaseModel):
    document_id: str
    content: Optional[Union[str, Dict[str, Any]]]
    metadata: Optional[Dict[str, Any]]
    status: str # SUCCESS/FAILURE/NOT_FOUND
    error_message: Optional[str] = None
```

### 4.3. Example API Method: `query_collection`
```python
from typing import List

class QueryCollectionInput(BaseModel):
    project_id: str
    collection_name: str
    query_texts: Optional[List[str]] = None # For semantic search
    where_filter: Optional[Dict[str, Any]] = None # For metadata filtering
    n_results: int = 10
    include: List[str] = ["metadatas", "documents"] # What to include in results

class QueryCollectionOutput(BaseModel):
    results: Optional[Dict[str, Any]] # ChromaDB query results structure
    status: str
    error_message: Optional[str] = None
```

## 5. API Contracts (Key Public Methods)

*(This agent will be implemented as a Python class with methods, not as an MCP tool itself. It will use `chungoid-core.utils.chroma_utils` or similar.)*

```python
from typing import Dict, Any, Union, Optional, List
# Assuming Pydantic models from section 4 are defined elsewhere (e.g., in pcma_schemas.py)

class IProjectChromaManagerAgent_v1:
    def __init__(self, base_chroma_path: str, project_id: str):
        """ Initializes the manager for a specific project. """
        raise NotImplementedError

    async def ensure_collection_exists(self, collection_name: str, embedding_function_name: Optional[str] = "default") -> bool:
        raise NotImplementedError

    async def store_artifact(
        self, 
        collection_name: str, 
        artifact_content: Union[str, Dict[str, Any]], 
        metadata: Dict[str, Any], 
        document_id: Optional[str] = None
    ) -> StoreArtifactOutput:
        """ Stores an artifact, handles embedding for string content. """
        raise NotImplementedError

    async def retrieve_artifact(
        self, 
        collection_name: str, 
        document_id: str
    ) -> RetrieveArtifactOutput:
        raise NotImplementedError

    async def update_artifact_metadata(
        self, 
        collection_name: str, 
        document_id: str, 
        new_metadata: Dict[str, Any]
    ) -> StoreArtifactOutput: # Similar output to store
        raise NotImplementedError

    async def query_collection_by_metadata(
        self, 
        collection_name: str, 
        where_filter: Dict[str, Any], 
        n_results: int = 10,
        include: List[str] = ["metadatas", "documents", "ids"]
    ) -> QueryCollectionOutput:
        raise NotImplementedError

    async def query_collection_by_text(
        self, 
        collection_name: str, 
        query_texts: List[str], 
        where_filter: Optional[Dict[str, Any]] = None, 
        n_results: int = 10,
        include: List[str] = ["metadatas", "documents", "ids", "distances"]
    ) -> QueryCollectionOutput:
        raise NotImplementedError

    async def delete_artifact(self, collection_name: str, document_id: str) -> StoreArtifactOutput:
        raise NotImplementedError
```

## 6. Key Algorithms & Logic Flows

### 6.1. Artifact Storage Logic
1.  Receive artifact content, target collection, metadata, and optional ID.
2.  Ensure the target collection exists (create if not, using appropriate embedding function for the collection type).
3.  If `document_id` is provided, check if it exists. Handle updates/overwrites based on policy (e.g., versioning or simple replacement).
4.  If content is string and collection is for text embeddings, generate embedding (e.g., using default sentence transformer from `chroma_utils`).
5.  Add/Upsert the document (content + embedding) and its metadata to the ChromaDB collection.
6.  Return result (ID, status).

### 6.2. Query Logic
1.  Receive collection name, query parameters (text, filter, n_results).
2.  Get the ChromaDB collection.
3.  Perform the query (e.g., `collection.query(...)`).
4.  Format and return results.

### 6.3. Project-Specific ChromaDB Initialization
*   When instantiated for a `project_id`, the PCMA might create a unique ChromaDB path or prefix for that project's collections (e.g., `main_chroma_db_path/project_id/collection_name`) to ensure isolation, or use a single DB with project_id in metadata for filtering if ChromaDB setup favors fewer DBs.
*   It will define and ensure the existence of the standard set of collections listed in section 1.2.1 and Section 8.3 of this document, each with an appropriate embedding function configuration if needed.

## 7. Prompting Strategy & Templates

*   Not applicable. This agent is a programmatic utility, not LLM-based.

## 8. Interaction with ChromaDB (via `chroma_utils` or similar)

### 8.1. ChromaDB Client Initialization
*   The agent will initialize a ChromaDB client (e.g., `chromadb.HttpClient` or `chromadb.PersistentClient`) pointing to the configured ChromaDB service or path.
*   It will likely use utility functions (e.g., from `chungoid-core.utils.chroma_utils`) to simplify client setup and embedding function management.

### 8.2. Collection Access
*   Uses `client.get_or_create_collection(name, embedding_function)`.
*   Specific embedding functions might be configured per collection type (e.g., default for general text, specialized for code, or `chromadb.utils.embedding_functions.OpenAIEmbeddingFunction` if using OpenAI embeddings and API key is available).

### 8.3. Managed Collections & Their Purpose (P3.M0.3.3)
1.  `project_goals`: (Default Text Embedding)
    *   Stores: `refined_user_goal.md`, `assumptions_and_ambiguities.md`.
    *   Purpose: Initial input and context for the project.
2.  `project_planning_artifacts`: (Default Text Embedding for MD/YAML, potentially No Embedding for JSON if primarily retrieved by ID)
    *   Stores: `llm_optimized_prd.json` (LOPRDs), `ProjectBlueprint.md`, `MasterExecutionPlan.yaml`.
    *   Purpose: Core planning and requirements documents.
3.  `schema_definitions`: (No Embedding - retrieved by ID/name)
    *   Stores: Definitions like `loprd_schema.json` (if versioned or project-specific).
    *   Purpose: To provide structure for other artifacts.
4.  `risk_assessment_reports`: (Default Text Embedding)
    *   Stores: Reports from `ProactiveRiskAssessorAgent` (e.g., `RiskAssessmentReport.md`).
    *   Purpose: Tracking project risks and mitigations.
5.  `traceability_reports`: (Default Text Embedding)
    *   Stores: Reports from `RequirementsTracerAgent` (e.g., `TraceabilityReport.md`).
    *   Purpose: Ensuring requirements coverage.
6.  `optimization_suggestion_reports`: (Default Text Embedding)
    *   Stores: Suggestions from various agents.
    *   Purpose: Tracking potential improvements.
7.  `live_codebase_collection`: (Code-Specific Embedding Function if available, else Default Text Embedding)
    *   Stores: Embeddings and content of generated/integrated code files.
    *   Purpose: RAG for code generation, understanding existing code.
8.  `library_documentation_collection`: (Default Text Embedding)
    *   Stores: Documentation for external libraries relevant to the project (fetched via MCP tools).
    *   Purpose: Context for code generation and problem-solving.
9.  `external_mcp_tools_documentation_collection`: (Default Text Embedding)
    *   Stores: Documentation for external MCP tools discovered.
    *   Purpose: Context for agent tool usage.
10. `agent_reflections_and_logs`: (Default Text Embedding or No Embedding if primarily filtered by metadata)
    *   Stores: Operational logs, decision rationale, errors, confidence scores from agents.
    *   Purpose: Auditing, debugging, learning, improving agent performance.
11. `test_reports_collection`: (Default Text Embedding for summaries, JSON for structured data)
    *   Stores: Reports from `SystemTestRunnerAgent`.
    *   Purpose: Tracking test outcomes.
12. `quality_assurance_logs`: (Default Text Embedding or No Embedding)
    *   Stores: Logs from ARCA, QA-related decisions, and artifact quality assessments.
    *   Purpose: Tracking overall quality process.

## 9. Confidence Score Generation & Interpretation

*   Not applicable for generation. This agent consumes confidence scores as metadata when storing artifacts from other agents but does not generate its own for its operations.

## 10. Error Handling, Resilience, and Retry Mechanisms

### 10.1. Error Detection & Reporting
*   Detects errors from ChromaDB client (e.g., connection errors, query failures, storage failures).
*   Input validation errors for API method parameters.
*   Errors will be returned in the `error_message` field of method outputs (e.g., `StoreArtifactOutput.error_message`).

### 10.2. Retry Strategies
*   May implement retries with backoff for transient ChromaDB connection issues for key operations (e.g., 1-2 retries).

### 10.3. Failure Escalation
*   Persistent failures are reported back to the calling agent via the `status` and `error_message` in the output of its API methods.
*   The calling agent is responsible for handling these failures.

### 10.4. Resilience to Input Issues
*   Method inputs will be validated using Pydantic models.
*   Handles cases like `document_not_found` gracefully by returning appropriate status.

## 11. Testing Strategy & Metrics

### 11.1. Unit Tests
*   Test each API method with mocked ChromaDB client interactions:
    *   Verify correct parameters are passed to the Chroma client.
    *   Verify correct handling of successful and error responses from the mock.
    *   Test logic for ID generation, metadata formatting, embedding calls (if any).
    *   Test `ensure_collection_exists` logic.

### 11.2. Integration Tests
*   Test against a live local ChromaDB instance.
*   Verify actual storage, retrieval, querying, and deletion of artifacts in various collections.
*   Test with different data types (text, JSON) and metadata.
*   Test semantic search functionality for relevant collections.

### 11.3. Performance Tests (If Applicable)
*   Measure average time for storing and retrieving artifacts of typical sizes.
*   Measure query times for common query patterns.

### 11.4. Key Metrics for Success/Evaluation
*   **Reliability:** Low failure rate for its operations.
*   **Ease of Use:** Clarity and effectiveness of its API for other agents.
*   **Performance:** Acceptable latency for CRUD and query operations.
*   **Data Integrity:** Ensures data is stored and retrieved accurately with correct metadata.

## 12. Alternatives Considered & Rationale for Chosen Design

*   **Alternative 1: Direct ChromaDB access by all agents:**
    *   **Pros:** Agents have full control.
    *   **Cons:** Inconsistent client initialization, duplicated logic for embeddings/metadata, harder to manage collection standards, tight coupling of all agents to ChromaDB specifics.
    *   **Reason for not choosing:** A centralized manager agent promotes separation of concerns, consistency, and easier maintenance.
*   **Alternative 2: Generic DataStoreAgent (not ChromaDB specific):**
    *   **Pros:** Could abstract away the specific vector DB provider.
    *   **Cons:** The current focus is on ChromaDB; building a fully generic DB abstraction layer is significantly more complex and not immediately required. Specific ChromaDB features (like semantic search) are desired.
    *   **Reason for not choosing:** Focus on leveraging ChromaDB effectively first. A more generic abstraction could be a future evolution if needed.

## 13. Open Issues & Future Work

*   **Issue 1:** Strategy for handling very large artifacts (chunking for embedding/storage if necessary).
*   **Issue 2:** Fine-grained error reporting and retry logic for ChromaDB operations.
*   **Future Enhancement 1:** Support for transactions or atomic operations if ChromaDB offers them and they are needed.
*   **Future Enhancement 2:** More sophisticated versioning of artifacts within ChromaDB.
*   **Future Enhancement 3:** Schema enforcement/validation for JSON artifacts before storage (beyond just LOPRD).
*   **Future Enhancement 4:** Asynchronous embedding generation if storing large volumes of text becomes a bottleneck.

## 9. Future Enhancements / Next Steps

The current MVP of PCMA is heavily mocked. Key areas for immediate and future development include:

*   **Full ChromaDB Integration:**
    *   Replace all mocked interactions with actual calls to a ChromaDB instance using `chromadb.PersistentClient` or `chromadb.HttpClient`.
    *   Implement the `ChromaDBManager` class (or similar) within `chungoid-core.utils.chroma_utils` robustly and have PCMA utilize it for all backend operations.
*   **Comprehensive Metadata Strategy:**
    *   Define and enforce a richer, standardized metadata schema for all stored artifacts. This should include:
        *   `artifact_type` (e.g., LOPRD, Blueprint, CodeFile, TestReport).
        *   `version` (semantic versioning or hash-based).
        *   `source_agent_id`, `source_agent_version`.
        *   `generating_task_id` (from `MasterExecutionPlan`).
        *   `lineage_ids` (linking to parent/source artifacts, e.g., LOPRD ID for a Blueprint).
        *   `timestamp_created`, `timestamp_modified`.
        *   `tags` or `labels` for categorization.
*   **Artifact Versioning:**
    *   Implement a clear strategy for versioning artifacts. This could involve:
        *   Storing multiple versions of a document ID, perhaps with a version number in metadata or as part of the ID itself.
        *   Using content hashing to detect changes and decide when a new version is warranted.
*   **Advanced Query Capabilities:**
    *   Develop more sophisticated query methods beyond simple ID retrieval and basic metadata filtering.
    *   Support for time-based queries (e.g., "get all LOPRDs created last week").
    *   Enable complex graph-like queries based on artifact lineage (e.g., "get all code files derived from Blueprint X").
*   **Content Hashing and Deduplication:**
    *   Implement content hashing (e.g., MD5, SHA256) for stored artifacts to facilitate integrity checks and potential deduplication.
*   **Transactionality and Error Handling:**
    *   Improve error handling for ChromaDB operations.
    *   Consider if/how to handle partial failures or ensure atomicity for sequences of operations where relevant (ChromaDB itself has certain atomicity guarantees per operation).
*   **Scalability and Performance:**
    *   As project data grows, monitor and optimize query performance.
    *   Evaluate ChromaDB configurations for optimal performance based on expected load.
*   **Security Considerations:**
    *   Ensure that if ChromaDB is run as a remote service, appropriate authentication and authorization mechanisms are in place (though this might be an infrastructure concern outside PCMA itself).
*   **Embedding Management:**
    *   More flexible management of embedding functions, potentially allowing different models per collection or even per document type within a collection if feasible.
    *   Strategy for re-embedding if embedding models are updated.

These enhancements will transition PCMA from a conceptual placeholder to a robust and essential component of the Autonomous Project Engine.

---
*This is a living document.*
*Last updated: YYYY-MM-DD by Gemini Assistant* 