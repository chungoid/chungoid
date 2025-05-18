import typer
from pathlib import Path
import yaml
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict, Any, Optional, Set
import logging
import datetime
import os
import hashlib
import json
from chromadb.utils import embedding_functions # type: ignore
from rich.logging import RichHandler
from rich.console import Console
from rich.pretty import pprint

# Configure logging
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
# Use RichHandler for better CLI logging
logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
logger.setLevel(logging.INFO)


app = typer.Typer(help="CLI tool to embed various structured and unstructured documents into ChromaDB for chungoid-core.")
console = Console()

# --- Constants ---
# When in chungoid-core/scripts, parents[1] is chungoid-core root.
CORE_PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = CORE_PROJECT_ROOT / ".chungoid"
EMBEDDED_LOG_FILENAME = "core_embedded_files.log"
DEFAULT_COLLECTION_PREFIX = "core_" # Changed from "meta_"


# --- Schemas (Simplified for validation) ---
# These are examples; for chungoid-core, new schemas might be defined or this made more generic.
# "thought" type is less relevant for chungoid-core directly.
REQUIRED_CORE_COMPONENT_FIELDS = [
    "component_id",
    "description",
    "version",
    "tags",
    "interface_schema" # Example field
]
# For embedding library/API documentation summaries or key concepts
REQUIRED_CORE_LIBRARY_INFO_FIELDS = ["library_name", "version", "summary", "key_apis", "tags"]


# --- Helper Functions ---


def load_yaml(path: Path) -> List[Dict[str, Any]]:
    """Loads YAML data from a file. Handles single or multiple documents."""
    if path.suffix.lower() not in ['.yaml', '.yml']:
        logger.error(f"Attempted to load non-YAML file as YAML: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f: # Added encoding
            data = list(yaml.safe_load_all(f))
            data = [item for item in data if item is not None]
        if not data:
            logger.warning(f"YAML file is empty or contains no valid documents: {path}")
            return []
        # Ensure it's a list of dicts
        valid_data = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    valid_data.append(item)
                # Handle case where safe_load_all might return a list containing a single list of dicts
                elif isinstance(item, list) and all(isinstance(sub_item, dict) for sub_item in item):
                    valid_data.extend(item)
                else:
                    logger.warning(f"Item {i} in YAML file {path} is not a dictionary and was skipped.")
            if not valid_data and data: # Original data was not empty but no dicts found
                 logger.error(f"No valid dictionary items found in YAML file {path}. Content type: {type(data[0]) if data else 'empty'}")
                 return []
            return valid_data
        elif isinstance(data, dict): # Should not happen with safe_load_all but as a safeguard
            return [data]
        else:
            logger.error(f"Unexpected YAML structure in {path}. Expected list of dictionaries, got {type(data)}.")
            return []

    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {path}: {e}")
        return []
    except FileNotFoundError:
        logger.error(f"YAML file not found: {path}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred loading YAML {path}: {e}")
        return []


def validate_schema(item: Dict[str, Any], schema_type: str, item_index: int, file_path: Path) -> bool:
    """Validates a single YAML item against the required fields for its type."""
    if schema_type == "core_component":
        required_fields = REQUIRED_CORE_COMPONENT_FIELDS
    elif schema_type == "core_library_info":
        required_fields = REQUIRED_CORE_LIBRARY_INFO_FIELDS
    # 'documentation' type does not have a fixed schema here, content is embedded directly.
    elif schema_type == "documentation":
        return True 
    else:
        logger.error(f"Invalid schema type specified for validation: {schema_type}")
        return False

    missing_fields = [field for field in required_fields if field not in item or item[field] is None or str(item[field]).strip() == ""]
    if missing_fields:
        logger.warning(f"Item {item_index+1} in {file_path} (type '{schema_type}') is missing/empty required fields: {missing_fields}")
        return False
    if "tags" in required_fields and not isinstance(item.get('tags'), list):
         logger.warning(f"Item {item_index+1} in {file_path} (type '{schema_type}') has invalid 'tags' field (must be a list).")
         return False
    return True

def get_embedding_model(model_name: str) -> SentenceTransformer:
    """Loads and returns a SentenceTransformer model."""
    logger.info(f"Loading sentence-transformer model: {model_name}")
    try:
        model = SentenceTransformer(model_name)
        logger.info(f"Successfully loaded model: {model_name}")
        return model
    except Exception as e:
        logger.error(f"Error loading SentenceTransformer model '{model_name}': {e}")
        raise typer.Exit(code=1)

def prepare_embedding_text(item: Dict[str, Any], fields_to_embed: List[str]) -> str:
    """Extracts and concatenates text from specified fields for embedding."""
    texts = []
    for field in fields_to_embed:
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            texts.append(value.strip())
        elif isinstance(value, list) and all(isinstance(x, str) for x in value): # Handle list of strings
            texts.append(" ".join(value).strip())
        elif value is not None and str(value).strip(): # Convert other types to string if not empty
            texts.append(str(value).strip())
            logger.debug(f"Field '{field}' converted to string for embedding text: {str(value)[:100]}...")
    return "\n\n".join(texts)


def get_chroma_client(host: str, port: int, path: Optional[Path] = None, client_type: str = "http") -> chromadb.HttpClient | chromadb.PersistentClient:
    """Gets a ChromaDB client instance (HTTP or Persistent)."""
    try:
        if client_type.lower() == "persistent" and path:
            logger.info(f"Initializing ChromaDB PersistentClient (path={path.resolve()})")
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory for PersistentClient: {path.resolve()}")
            client = chromadb.PersistentClient(path=str(path.resolve()))
        elif client_type.lower() == "http":
            logger.info(f"Initializing ChromaDB HttpClient (host={host}, port={port})")
            client = chromadb.HttpClient(host=host, port=port)
        else:
            logger.error(f"Invalid client_type '{client_type}'. Must be 'http' or 'persistent'.")
            raise typer.Exit(code=1)
        
        client.heartbeat() # Verify connection
        logger.info(f"ChromaDB {client_type.capitalize()}Client connection successful.")
        return client
    except Exception as e:
        details = f"path={path.resolve() if path else 'N/A'}" if client_type == "persistent" else f"host={host}, port={port}"
        logger.error(f"Error connecting to ChromaDB ({details}): {e}")
        raise typer.Exit(code=1)

def load_embedded_log(log_file_path: Path) -> set:
    """Loads the set of already embedded file paths from the specified log file."""
    if not log_file_path.parent.exists():
         logger.warning(f"Log directory {log_file_path.parent} does not exist. Attempting to create.")
         try:
             log_file_path.parent.mkdir(parents=True, exist_ok=True)
         except Exception as e:
             logger.error(f"Failed to create log directory {log_file_path.parent}: {e}. Proceeding without persistent log tracking for this run.")
             return set() # Return empty set if dir creation fails, effectively disabling log check for this run

    if not log_file_path.exists():
        logger.info(f"Embedded log file not found at {log_file_path}. A new one will be created if items are embedded.")
        return set()
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f: # Added encoding
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        logger.warning(f"Could not read embedded log file {log_file_path}: {e}. Assuming no files logged.")
        return set()

def log_embedded_file(log_file_path: Path, file_path_to_log: Path):
    """Appends a file path to the embedded log."""
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f: # Added encoding
            f.write(f"{file_path_to_log.resolve()}\n") # Log resolved absolute path
    except Exception as e:
        logger.error(f"Error writing to embedded log file {log_file_path}: {e}")


# --- Typer Commands ---


@app.command("add")
def add_items(
    input_path: Path = typer.Argument(..., help="Path to the input file or directory.", exists=True, readable=True),
    type: str = typer.Option(..., "--type", "-t", help="Type of data ('core_component', 'core_library_info', or 'documentation').", case_sensitive=False, ),
    tags: Optional[List[str]] = typer.Option(None, "--tags", help="Comma-separated tags (used for 'documentation', or as default for YAML types if not in YAML)."),
    embed_fields: Optional[List[str]] = typer.Option(None, "--embed-fields", help="YAML fields to concatenate for embedding (ignored for 'documentation'). E.g., 'summary,details'"),
    id_field: Optional[str] = typer.Option(None, "--id-field", help="YAML field to use for Chroma document ID (ignored for 'documentation'). Fallback to auto-generated ID."),
    collection_prefix: str = typer.Option(DEFAULT_COLLECTION_PREFIX, "--collection-prefix", help="Prefix for Chroma collection names if not overridden."),
    collection_name_override: Optional[str] = typer.Option(None, "--collection", "-c", help="Explicitly specify the target collection name (overrides prefix logic)."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type: 'http' or 'persistent'."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname of the ChromaDB server (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port of the ChromaDB server (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path for ChromaDB (for 'persistent' client). Also used for log file dir."),
    embedding_model_name: str = typer.Option("all-MiniLM-L6-v2", "--embedding-model", help="Name of the sentence-transformer model."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Simulate actions without writing to ChromaDB."),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing of files already in the log."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """Embed various types of documents (YAML components, library info, or raw text documentation) into ChromaDB for chungoid-core."""

    if verbose:
        logger.setLevel(logging.DEBUG)
        logger.info("Verbose mode enabled.")

    log_file_actual_path = chroma_path.parent / EMBEDDED_LOG_FILENAME 
    # If chroma_path is /foo/bar/chroma_db, log is /foo/bar/core_embedded_files.log
    # If chroma_path is just /foo/bar (intending to be .chungoid), then parent is /foo, log is /foo/core_embedded_files.log
    # For consistency, let's ensure the log is placed inside a .chungoid-like directory structure
    # If chroma_path itself is intended to be the .chungoid directory:
    if chroma_path.name == "chroma_db" and chroma_path.parent.name == ".chungoid":
        log_file_actual_path = chroma_path.parent / EMBEDDED_LOG_FILENAME
    elif chroma_path.name == ".chungoid": # User specified .chungoid directly
        log_file_actual_path = chroma_path / EMBEDDED_LOG_FILENAME
    else: # Default behavior or other custom path, place log inside it or its parent
        # Safest is to ensure a .chungoid directory exists at the project root and place log there
        # Or simply place it next to the chroma_path if persistent, or in CORE_PROJECT_ROOT/.chungoid if http client
        # For now, using chroma_path.parent for the log file assumes chroma_path is like project_root/.chungoid/chroma_db
        # Let's make it more robust:
        if chroma_client_type == "persistent":
             # If persistent, log goes into the parent of the chroma_db directory itself.
             # Example: if chroma_path is project/.chungoid/chroma_db, log is project/.chungoid/core_embedded_files.log
            log_file_actual_path = chroma_path.parent / EMBEDDED_LOG_FILENAME
        else: # For HTTP client, place log in a default location within the core project
            log_file_actual_path = DEFAULT_LOG_DIR / EMBEDDED_LOG_FILENAME
            if not DEFAULT_LOG_DIR.exists(): DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Using log file at: {log_file_actual_path}")


    supported_types = ["core_component", "core_library_info", "documentation"]
    if type not in supported_types:
        logger.error(f"Invalid type specified. Must be one of: {', '.join(supported_types)}")
        raise typer.Exit(code=1)

    processed_items_data = []
    collection_map: Dict[str, List[Dict[str, Any]]] = {}
    files_to_process: List[Path] = []
    
    embedded_files_log_content = load_embedded_log(log_file_actual_path)
    logger.debug(f"Loaded {len(embedded_files_log_content)} entries from {log_file_actual_path}")

    # --- Determine files to process ---
    if input_path.is_file():
        files_to_process.append(input_path)
    elif input_path.is_dir():
        if type == "documentation":
            # For documentation, if a dir is given, process all .txt, .md files recursively
            # This is a change from original, which only allowed single file for docs.
            logger.info(f"Processing all .md and .txt files in directory for documentation type: {input_path}")
            files_to_process.extend(list(input_path.rglob("*.md")))
            files_to_process.extend(list(input_path.rglob("*.txt")))
        elif type in ["core_component", "core_library_info"]:
            logger.info(f"Processing all .yaml and .yml files in directory for type '{type}': {input_path}")
            files_to_process.extend(list(input_path.rglob("*.yaml")))
            files_to_process.extend(list(input_path.rglob("*.yml")))
        else:
            logger.error(f"Directory input is only supported for types 'documentation', 'core_component', or 'core_library_info'. Type given: {type}")
            raise typer.Exit(code=1)
    else:
        logger.error(f"Input path {input_path} is neither a file nor a directory.")
        raise typer.Exit(code=1)

    if not files_to_process:
        logger.info(f"No files found to process at {input_path} for type '{type}'.")
        raise typer.Exit()

    final_files_to_process: List[Path] = []
    for fp in files_to_process:
        abs_path_str = str(fp.resolve())
        if not force and abs_path_str in embedded_files_log_content:
            logger.info(f"Skipping already processed file (use --force to override): {fp.relative_to(Path.cwd())}")
            continue
        final_files_to_process.append(fp)

    if not final_files_to_process:
        logger.info("No new files to process after checking log and --force flag.")
        raise typer.Exit()
    
    logger.info(f"Found {len(final_files_to_process)} file(s) to process.")

    # --- Load Model ---
    if not dry_run:
        model = get_embedding_model(embedding_model_name)

    # --- Process each file ---
    for file_path in final_files_to_process:
        logger.info(f"Processing file: {file_path.relative_to(Path.cwd())}")
        if type == "documentation":
            try:
                content = file_path.read_text(encoding='utf-8') # Added encoding
                if not content.strip():
                    logger.warning(f"Documentation file is empty or contains only whitespace: {file_path}. Skipping.")
                    continue
                
                # Generate a stable ID based on the relative path to CWD to avoid issues if CWD changes
                # Using absolute path for uniqueness across potential different CWDs during runs
                doc_id = hashlib.sha1(str(file_path.resolve()).encode()).hexdigest()
                
                target_collection_name = collection_name_override if collection_name_override else f"{collection_prefix}documentation"
                
                item_data = {
                    "doc_id": doc_id,
                    "text_to_embed": content,
                    "metadata": {
                        "source_file": str(file_path.resolve()), 
                        "type": "documentation",
                        "original_filename": file_path.name,
                        "processed_timestamp": datetime.datetime.now().isoformat(),
                        "tags": tags or [] # Apply CLI tags if provided
                    },
                    "collection": target_collection_name,
                    "original_file_path": file_path # For logging
                }
                collection_map.setdefault(target_collection_name, []).append(item_data)
                processed_items_data.append(item_data)

            except Exception as e:
                logger.error(f"Error reading documentation file {file_path}: {e}")
                continue # Skip to next file
        
        elif type in ["core_component", "core_library_info"]:
            yaml_items = load_yaml(file_path)
            if not yaml_items:
                logger.warning(f"No valid YAML items loaded from {file_path}. Skipping.")
                continue

            default_embed_fields_map = {
                "core_component": ["component_id", "description", "tags"], # Example
                "core_library_info": ["library_name", "version", "summary", "key_apis", "tags"]
            }
            current_embed_fields = embed_fields if embed_fields else default_embed_fields_map.get(type, [])
            if not current_embed_fields:
                 logger.error(f"No embed_fields defined for type '{type}'. Please specify --embed-fields or update defaults. Skipping {file_path}.")
                 continue
            
            logger.debug(f"Using embed_fields for type '{type}': {current_embed_fields}")

            for i, item in enumerate(yaml_items):
                if not validate_schema(item, type, i, file_path):
                    logger.warning(f"Schema validation failed for item {i+1} in {file_path}. Skipping item.")
                    continue
                
                text_to_embed = prepare_embedding_text(item, current_embed_fields)
                if not text_to_embed.strip():
                    logger.warning(f"No text content to embed for item {i+1} in {file_path} after preparing fields: {current_embed_fields}. Skipping item.")
                    continue
                
                item_id_value = None
                if id_field and item.get(id_field):
                    item_id_value = str(item[id_field])
                else: # Auto-generate ID
                    # Use hash of file path + item index + critical field content for more stability if item moves within file
                    critical_content_hash = hashlib.sha1(text_to_embed[:500].encode()).hexdigest()[:8]
                    item_id_value = f"{file_path.stem}_{i}_{critical_content_hash}"
                
                target_collection_name = collection_name_override if collection_name_override else f"{collection_prefix}{type}"
                
                # Prepare metadata, ensure all values are suitable for Chroma (str, int, float, bool)
                metadata = {
                    "source_file": str(file_path.resolve()),
                    "type": type,
                    "original_filename": file_path.name,
                    "item_index_in_file": i,
                    "processed_timestamp": datetime.datetime.now().isoformat(),
                }
                # Add all string/numeric/bool fields from item to metadata, excluding complex objects/lists unless handled
                for k, v in item.items():
                    if isinstance(v, (str, int, float, bool)):
                        metadata[k] = v
                    elif isinstance(v, list) and all(isinstance(x, (str, int, float, bool)) for x in v):
                        metadata[k] = v # Chroma supports lists of primitive types
                    elif k == "tags" and isinstance(v, list): # Ensure tags from item are lists of strings
                        metadata[k] = [str(tag) for tag in v if isinstance(tag, (str, int, float, bool))]
                
                # Apply CLI tags if no tags in item, or add to existing if item_tags is a list
                cli_tags_to_apply = tags or []
                if 'tags' in metadata and isinstance(metadata['tags'], list):
                    metadata['tags'].extend(t for t in cli_tags_to_apply if t not in metadata['tags'])
                elif cli_tags_to_apply: # if item had no tags or not a list
                    metadata['tags'] = cli_tags_to_apply
                
                item_data = {
                    "doc_id": item_id_value,
                    "text_to_embed": text_to_embed,
                    "metadata": metadata,
                    "collection": target_collection_name,
                    "original_file_path": file_path # For logging
                }
                collection_map.setdefault(target_collection_name, []).append(item_data)
                processed_items_data.append(item_data)
    
    if not processed_items_data:
        logger.info("No items were processed successfully for embedding.")
        raise typer.Exit()

    logger.info(f"Prepared {len(processed_items_data)} items for embedding across {len(collection_map)} collection(s).")
    if dry_run:
        console.print("\n[bold yellow]Dry Run Mode - Printing items that would be embedded:[/bold yellow]")
        for coll_name, items_in_coll in collection_map.items():
            console.print(f"\n--- Collection: [bold cyan]{coll_name}[/bold cyan] ({len(items_in_coll)} items) ---")
            for item_to_embed in items_in_coll:
                console.print(f"  ID: {item_to_embed['doc_id']}")
                console.print(f"  Metadata: {item_to_embed['metadata']}")
                console.print(f"  Text (first 100 chars): {item_to_embed['text_to_embed'][:100].replace('\n', ' ')}...")
                console.print(f"  (Logged to: {item_to_embed['original_file_path'].name})") # Show which file was logged
        logger.info("Dry run complete. No changes made to ChromaDB or log file.")
        raise typer.Exit()

    # --- Actual Embedding ---    
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_model_name,_model=model)

    total_embedded_count = 0
    for coll_name, items_in_coll in collection_map.items():
        logger.info(f"Embedding {len(items_in_coll)} items into collection: {coll_name}")
        try:
            collection = client.get_or_create_collection(
                name=coll_name,
                embedding_function=embedding_function # type: ignore
            )
            
            ids_batch = [item['doc_id'] for item in items_in_coll]
            documents_batch = [item['text_to_embed'] for item in items_in_coll]
            metadatas_batch = [item['metadata'] for item in items_in_coll]
            
            # ChromaDB add/upsert can take lists directly
            collection.upsert(
                ids=ids_batch,
                documents=documents_batch,
                metadatas=metadatas_batch
            )
            logger.info(f"Successfully upserted {len(ids_batch)} items into '{coll_name}'.")
            total_embedded_count += len(ids_batch)

            # Log successfully embedded files (original paths)
            for item_logged in items_in_coll:
                log_embedded_file(log_file_actual_path, item_logged['original_file_path'])

        except Exception as e:
            logger.error(f"Error embedding items into collection '{coll_name}': {e}", exc_info=True)
            # Decide if we should continue with other collections or exit
            # For now, continue to try embedding other collections

    logger.info(f"Embedding process complete. Total items embedded in this run: {total_embedded_count}")
    if total_embedded_count < len(processed_items_data):
        logger.warning(f"{len(processed_items_data) - total_embedded_count} items failed to embed due to errors in specific collections. Check logs above.")

# --- Other CLI Commands (List, Get, Query, etc.) --- 
# These are mostly for inspection and management, adapting them lightly.

@app.command("list-collections")
def list_collections_command(
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type: 'http' or 'persistent'."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client).")
):
    """List all collections in the ChromaDB instance."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    logger.info("Fetching collections...")
    collections = client.list_collections()
    if collections:
        console.print("[bold green]Available collections:[/bold green]")
        for coll in collections:
            # console.print(f"  - Name: {coll.name}, ID: {coll.id}, Metadata: {coll.metadata}, Count: {coll.count()}")
            # ChromaDB 0.4+ Collection object structure:
            console.print(f"  - Name: {coll.name}, ID: {coll.id}")
            try:
                console.print(f"    Metadata: {coll.metadata}") # May be None
                console.print(f"    Count: {coll.count()}")
            except Exception as e:
                logger.warning(f"Could not retrieve full details for {coll.name}: {e}")

    else:
        console.print("[yellow]No collections found.[/yellow]")

# Helper for parsing include/where strings if needed (copied from original, may need adjustment)
def _parse_include_string(include_str: Optional[str], default: List[str]) -> Optional[List[str]]:
    if include_str is None:
        return default
    return [field.strip() for field in include_str.split(',') if field.strip()]

def _parse_where_string(where_str: Optional[str]) -> Optional[Dict[str, Any]]:
    if where_str is None:
        return None
    try:
        return json.loads(where_str)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in --where string: {e}")
        raise typer.Exit(code=1)

@app.command("list-items")
def list_items_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection to list items from."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client)."),
    limit: int = typer.Option(10, "--limit", "-n", help="Maximum number of items to return."),
    offset: int = typer.Option(0, "--offset", help="Number of items to skip."),
    include: Optional[str] = typer.Option("metadatas,documents", "--include", help="Comma-separated fields: metadatas, documents, embeddings."),
    where: Optional[str] = typer.Option(None, "--where", help="JSON string for filtering metadata e.g., '{\"type\":\"documentation\"}'.")
):
    """List items from a specified collection."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        include_list = _parse_include_string(include, default=["metadatas", "documents"])
        where_dict = _parse_where_string(where)
        
        logger.info(f"Fetching items from '{collection_name}' (limit={limit}, offset={offset}, where={where_dict}, include={include_list})...")
        results = collection.get(limit=limit, offset=offset, where=where_dict, include=include_list) # type: ignore
        
        console.print(f"[bold green]Items in '{collection_name}':[/bold green]")
        # Determine what fields are present in results to avoid KeyErrors
        num_items = len(results.get('ids', []))
        if num_items == 0:
            console.print("  No items found matching criteria.")
            return

        for i in range(num_items):
            console.print(f"  --- Item {offset + i + 1} ---")
            if results.get('ids') and i < len(results['ids']):
                 console.print(f"    ID: {results['ids'][i]}")
            if results.get('metadatas') and results['metadatas'] and i < len(results['metadatas']):
                 console.print(f"    Metadata: {results['metadatas'][i]}")
            if results.get('documents') and results['documents'] and i < len(results['documents']):
                 console.print(f"    Document (first 100 chars): {results['documents'][i][:100].replace('\n', ' ')}...")
            if results.get('embeddings') and results['embeddings'] and i < len(results['embeddings']):
                 console.print(f"    Embedding (first 5 values): {results['embeddings'][i][:5]}...")

    except Exception as e:
        logger.error(f"Error listing items from collection '{collection_name}': {e}", exc_info=True)

@app.command("get-item")
def get_item_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection."),
    item_id: str = typer.Argument(..., help="ID of the item to retrieve."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client)."),
    include: Optional[str] = typer.Option("metadatas,documents,embeddings", "--include", help="Comma-separated fields to include.")
):
    """Retrieve a specific item by its ID from a collection."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        include_list = _parse_include_string(include, default=["metadatas", "documents", "embeddings"])
        logger.info(f"Fetching item '{item_id}' from '{collection_name}' (include={include_list})...")
        result = collection.get(ids=[item_id], include=include_list) # type: ignore

        if not result.get('ids') or not result['ids'][0]:
            console.print(f"[yellow]Item with ID '{item_id}' not found in collection '{collection_name}'.[/yellow]")
            return

        console.print(f"[bold green]Item '{item_id}' from '{collection_name}':[/bold green]")
        if result.get('ids') and result['ids'][0]: # Should always be true if found
             console.print(f"  ID: {result['ids'][0]}")
        if result.get('metadatas') and result['metadatas']:
             console.print(f"  Metadata: {result['metadatas'][0]}")
        if result.get('documents') and result['documents']:
             console.print(f"  Document: {result['documents'][0]}")
        if result.get('embeddings') and result['embeddings']:
             console.print(f"  Embedding (first 10 values): {result['embeddings'][0][:10]}...")

    except Exception as e:
        logger.error(f"Error getting item '{item_id}' from '{collection_name}': {e}", exc_info=True)

@app.command("query")
def query_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection to query."),
    query_text: str = typer.Argument(..., help="The text to query for similarity."),
    n_results: int = typer.Option(5, "--n-results", "-n", help="Number of similar results to return."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client)."),
    where: Optional[str] = typer.Option(None, "--where", help="JSON string for filtering metadata."),
    include: Optional[str] = typer.Option("metadatas,documents,distances", "--include", help="Comma-separated fields to include.")
):
    """Perform a similarity query against a collection."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        include_list = _parse_include_string(include, default=["metadatas", "documents", "distances"])
        where_dict = _parse_where_string(where)

        logger.info(f"Querying '{collection_name}' with text: '{query_text[:50]}...' (n_results={n_results}, where={where_dict}, include={include_list})")
        results = collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where_dict, # type: ignore
            include=include_list # type: ignore
        )
        console.print(f"[bold green]Query results from '{collection_name}' for '{query_text[:50]}...':[/bold green]")
        num_results_found = len(results.get('ids', [[]])[0]) # Query returns list of lists for ids, docs etc.

        if num_results_found == 0:
            console.print("  No results found.")
            return

        for i in range(num_results_found):
            console.print(f"  --- Result {i+1} ---")
            if results.get('ids') and results['ids'][0] and i < len(results['ids'][0]):
                console.print(f"    ID: {results['ids'][0][i]}")
            if results.get('distances') and results['distances'][0] and i < len(results['distances'][0]):
                console.print(f"    Distance: {results['distances'][0][i]:.4f}")
            if results.get('metadatas') and results['metadatas'][0] and i < len(results['metadatas'][0]):
                console.print(f"    Metadata: {results['metadatas'][0][i]}")
            if results.get('documents') and results['documents'][0] and i < len(results['documents'][0]):
                console.print(f"    Document (first 100 chars): {results['documents'][0][i][:100].replace('\n', ' ')}...")
            if results.get('embeddings') and results['embeddings'][0] and i < len(results['embeddings'][0]):
                 console.print(f"    Embedding (first 5 values): {results['embeddings'][0][i][:5]}...")
                
    except Exception as e:
        logger.error(f"Error querying collection '{collection_name}': {e}", exc_info=True)

@app.command("count")
def count_items_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection to count items in."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client).")
):
    """Count the total number of items in a collection."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        console.print(f"Collection '{collection_name}' contains [bold cyan]{count}[/bold cyan] items.")
    except Exception as e:
        logger.error(f"Error counting items in '{collection_name}': {e}", exc_info=True)

@app.command("delete-items")
def delete_items_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection to delete items from."),
    item_ids_str: str = typer.Argument(..., help="Comma-separated list of item IDs to delete. Use '@all' to attempt to delete all items (requires confirmation)."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client).")
):
    """Delete specified items (by ID) from a collection, or all items if '@all' is specified."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        if item_ids_str.lower() == "@all":
            current_count = collection.count()
            if not typer.confirm(f"Are you sure you want to delete all {current_count} items from collection '{collection_name}'? This cannot be undone."):
                console.print("Deletion cancelled.")
                raise typer.Exit()
            logger.info(f"Attempting to delete all items from '{collection_name}'...")
            # To delete all, pass no IDs or where clause to delete() (ChromaDB API behavior)
            # However, some versions of ChromaDB client might not support collection.delete() with no args directly for all items.
            # A safer way for "all" might be to get all IDs and pass them, or use client.delete_collection if that's the intent.
            # For now, let's assume we mean deleting items based on some criteria, or specific IDs.
            # If truly all items, it's usually `client.delete_collection` and then `client.create_collection`.
            # Let's implement delete by specific IDs for now, and "@all" can be enhanced if needed.
            console.print("Deleting all items via individual ID fetching is not implemented. Consider deleting and recreating the collection for this.")
            logger.warning("'@all' for item deletion is not fully supported by this command yet. Please delete specific IDs or manage the collection directly.")
            raise typer.Exit(code=1)
        else:
            item_ids_list = [id_val.strip() for id_val in item_ids_str.split(',') if id_val.strip()]
            if not item_ids_list:
                logger.error("No item IDs provided for deletion.")
                raise typer.Exit(code=1)
            logger.info(f"Attempting to delete items with IDs: {item_ids_list} from '{collection_name}'...")
            collection.delete(ids=item_ids_list)
            console.print(f"Successfully initiated deletion of items: {item_ids_list} from '{collection_name}'. Verify count if needed.")

    except Exception as e:
        logger.error(f"Error deleting items from '{collection_name}': {e}", exc_info=True)

@app.command("peek")
def peek_collection_command(
    collection_name: str = typer.Option(..., "--collection", "-c", help="Name of the collection to peek into."),
    n_results: int = typer.Option(5, "--n-results", "-n", help="Number of items to retrieve."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client).")
):
    """Retrieve a few items from a collection to get a sense of its content."""
    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name)
        results = collection.peek(limit=n_results)
        console.print(f"[bold green]Peeking into '{collection_name}' (first {len(results.get('ids',[]))} items):[/bold green]")
        # Similar display logic as list-items
        num_items = len(results.get('ids', []))
        if num_items == 0:
            console.print("  Collection appears to be empty or peek returned no items.")
            return

        for i in range(num_items):
            console.print(f"  --- Item {i + 1} ---")
            if results.get('ids') and i < len(results['ids']):
                 console.print(f"    ID: {results['ids'][i]}")
            if results.get('metadatas') and results['metadatas'] and i < len(results['metadatas']):
                 console.print(f"    Metadata: {results['metadatas'][i]}")
            if results.get('documents') and results['documents'] and i < len(results['documents']):
                 console.print(f"    Document (first 100 chars): {results['documents'][i][:100].replace('\n', ' ')}...")
            if results.get('embeddings') and results['embeddings'] and i < len(results['embeddings']):
                 console.print(f"    Embedding (first 5 values): {results['embeddings'][i][:5]}...")

    except Exception as e:
        logger.error(f"Error peeking into collection '{collection_name}': {e}", exc_info=True)

@app.command("validate-log")
def validate_log_command(
    collection_name_to_check: str = typer.Option(..., "--collection", "-c", help="Name of the collection to validate log entries against."),
    chroma_client_type: str = typer.Option("persistent", "--chroma-client-type", help="Chroma client type."),
    chroma_host: str = typer.Option("localhost", "--chroma-host", help="Hostname (for 'http' client)."),
    chroma_port: int = typer.Option(8000, "--chroma-port", help="Port (for 'http' client)."),
    chroma_path: Path = typer.Option(DEFAULT_LOG_DIR / "chroma_db", "--chroma-path", help="Filesystem path (for 'persistent' client)."),
    log_path_override: Optional[Path] = typer.Option(None, "--log-path", help="Override default log file path to validate."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
):
    """Validate that all files listed in the log exist in the specified ChromaDB collection."""
    if verbose:
        logger.setLevel(logging.DEBUG)

    actual_log_file = log_path_override if log_path_override else (chroma_path.parent / EMBEDDED_LOG_FILENAME if chroma_client_type == "persistent" else DEFAULT_LOG_DIR / EMBEDDED_LOG_FILENAME)
    logger.info(f"Validating log file {actual_log_file} against collection '{collection_name_to_check}'")

    logged_file_paths_str = load_embedded_log(actual_log_file)
    if not logged_file_paths_str:
        console.print(f"[yellow]Log file {actual_log_file} is empty or unreadable. No validation possible.[/yellow]")
        return

    client = get_chroma_client(chroma_host, chroma_port, chroma_path, chroma_client_type)
    try:
        collection = client.get_collection(name=collection_name_to_check)
    except Exception as e:
        logger.error(f"Could not get collection '{collection_name_to_check}' from ChromaDB: {e}. Cannot validate log.")
        raise typer.Exit(code=1)

    missing_in_chroma_count = 0
    found_in_chroma_count = 0
    # We need to check if documents originating from these logged file paths are in Chroma.
    # The ID generation for type 'documentation' is sha1(resolved_path).
    # For YAML types, IDs are more complex (e.g., filestem_index_contenthash).
    # This validation is simpler for 'documentation' type if IDs are consistent.
    # For now, we'll just check if ANY document with 'source_file' metadata matching a logged path exists.
    # This is an approximation.
    logger.info(f"Checking {len(logged_file_paths_str)} logged file paths...")
    
    all_collection_docs = collection.get(include=["metadatas"]) # Get all metadatas
    chroma_source_files = set()
    if all_collection_docs.get('metadatas'):
        for meta in all_collection_docs['metadatas']:
            if meta and isinstance(meta, dict) and meta.get('source_file'):
                chroma_source_files.add(str(Path(meta['source_file']).resolve()))

    logger.debug(f"Found {len(chroma_source_files)} unique source_file entries in Chroma collection '{collection_name_to_check}'.")

    for logged_path_str in logged_file_paths_str:
        resolved_logged_path = str(Path(logged_path_str).resolve()) # Ensure comparison with resolved paths
        if resolved_logged_path in chroma_source_files:
            found_in_chroma_count +=1
            if verbose:
                logger.debug(f"OK: Logged file {resolved_logged_path} appears to have corresponding entry in Chroma.")
        else:
            missing_in_chroma_count += 1
            logger.warning(f"MISSING?: Logged file {resolved_logged_path} does not have a direct match for 'source_file' in Chroma collection '{collection_name_to_check}'.")
    
    console.print(f"\nValidation Summary for log '{actual_log_file}' against collection '{collection_name_to_check}':")
    console.print(f"  - Logged file entries: {len(logged_file_paths_str)}")
    console.print(f"  - Entries found with matching 'source_file' in Chroma: [green]{found_in_chroma_count}[/green]")
    console.print(f"  - Entries potentially missing or with different 'source_file' in Chroma: [yellow]{missing_in_chroma_count}[/yellow]")

    if missing_in_chroma_count > 0:
        logger.warning("Validation found potential discrepancies. This could be due to ID generation differences for YAML items, or files truly missing. Manual review may be needed.")
    else:
        console.print("[bold green]Log validation complete. All logged files appear to have corresponding entries in Chroma based on 'source_file' metadata.[/bold green]")


if __name__ == "__main__":
    app()

# Placeholder for further schema definitions or complex logic if needed.
# Example of a more complex thought schema if validation were deeper:
# THOUGHT_SCHEMA = {
#     "type": "object",
#     "properties": {
#         "id": {"type": "string", "format": "uuid"},
#         "timestamp": {"type": "string", "format": "date-time"},
#         "author": {"type": "string", "minLength": 1},
#         # ... other fields
#     },
#     "required": ["id", "timestamp", "author", ...]
# } 