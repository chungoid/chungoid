import chromadb
import yaml
from pathlib import Path

COLLECTION_NAME = "chungoid_context"
CONFIG_FILE = "config.yaml"

print("--- ChromaDB Direct Check ---")

# 1. Read Config
config_path = Path(CONFIG_FILE)
if not config_path.is_file():
    print(f"ERROR: Configuration file not found at {config_path.resolve()}")
    exit(1)

try:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    chroma_config = config.get("chromadb", {})
    client_type = chroma_config.get("client_type", "unknown").lower()
    host = chroma_config.get("host", "localhost")
    port = chroma_config.get("port", 8000)
    print(f"Read config: client_type='{client_type}', host='{host}', port={port}")
except Exception as e:
    print(f"ERROR: Failed to read or parse {CONFIG_FILE}: {e}")
    exit(1)

# 2. Check Client Type and Connect
if client_type != "http":
    print(f"ERROR: Script only supports 'http' client_type. Found '{client_type}' in config.")
    exit(1)

try:
    print(f"Attempting to connect to ChromaDB server at http://{host}:{port}...")
    client = chromadb.HttpClient(host=host, port=port)
    # Optional: Check server heartbeat/version if needed and available
    # print(f"Server version: {client.version()}")
    print("Connection successful.")
except Exception as e:
    print(f"ERROR: Failed to connect to ChromaDB server: {e}")
    exit(1)

# 3. Check Collection
try:
    print(f"Attempting to get collection: '{COLLECTION_NAME}'...")
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"Collection '{COLLECTION_NAME}' found.")
    count = collection.count()
    print(f"Current item count in collection: {count}")
except Exception as e:
    print(f"INFO: Could not get collection '{COLLECTION_NAME}'. It might not exist yet. Error: {e}")
    print("Attempting to create the collection...")
    try:
        collection = client.create_collection(name=COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' created successfully.")
        count = collection.count()
        print(f"Current item count in new collection: {count}")
    except Exception as create_err:
        print(f"ERROR: Failed to create collection '{COLLECTION_NAME}': {create_err}")
        exit(1)


# 4. Test Add Operation
test_doc_id = "direct_test_doc_1"
test_doc_content = "This is a direct test document added via script."
test_doc_metadata = {"source": "check_chroma.py", "status": "testing"}
try:
    print(f"Attempting to add test document with id: '{test_doc_id}'...")
    collection.add(ids=[test_doc_id], documents=[test_doc_content], metadatas=[test_doc_metadata])
    print("Test document added successfully.")
except Exception as e:
    print(f"ERROR: Failed to add test document: {e}")
    # Continue to query attempt anyway

# 5. Test Query Operation
try:
    print(f"Attempting to query for test document content: '{test_doc_content[:20]}...'")
    results = collection.query(
        query_texts=[test_doc_content],
        n_results=1,
        where={"source": "check_chroma.py"},  # Filter for our test doc
    )
    print(f"Query results: {results}")
    if results and results.get("ids") and results["ids"][0] and results["ids"][0][0] == test_doc_id:
        print("SUCCESS: Test document successfully added and retrieved.")
    else:
        print("FAILURE: Test document was not found after adding, or query failed.")
except Exception as e:
    print(f"ERROR: Failed to query collection: {e}")

print("--- Check Complete ---") 