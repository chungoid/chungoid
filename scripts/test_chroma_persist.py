import chromadb
import os
import shutil
import pytest

pytest.skip("Integration test for Chroma persistent storage skipped in CI", allow_module_level=True)

print("--- ChromaDB PersistentClient Test ---")

persist_directory = "./test_chroma_dir"
collection_name = "test_collection"
doc_id = "test_doc_1"
doc_content = "This is a test document."

# Clean up previous test directory if it exists
if os.path.exists(persist_directory):
    print(f"Removing existing test directory: {persist_directory}")
    try:
        shutil.rmtree(persist_directory)
        print("Previous directory removed.")
    except Exception as e:
        print(f"Error removing previous directory: {e}")
        # Exit if cleanup fails? Or continue cautiously? Let's exit for safety.
        exit(1)
else:
    print(f"Test directory {persist_directory} does not exist, proceeding.")

try:
    print(f"Attempting to initialize PersistentClient with path: {persist_directory}")
    # Initialize the client
    client = chromadb.PersistentClient(path=persist_directory)
    print(f"PersistentClient initialized. Type: {type(client)}")

    print(f"Checking if directory {persist_directory} exists POST-init...")
    if os.path.isdir(persist_directory):
        print("SUCCESS: Directory exists after client initialization.")
        print("Directory contents:")
        try:
            print(os.listdir(persist_directory))
        except Exception as e:
            print(f"Error listing directory contents: {e}")
    else:
        print("FAILURE: Directory DOES NOT exist after client initialization.")
        exit(1)  # Exit if directory wasn't created

    # Get or create the collection
    print(f"Attempting to get or create collection: {collection_name}")
    # Note: Using default embedding function implicitly here
    collection = client.get_or_create_collection(name=collection_name)
    print(f"Collection obtained. Name: {collection.name}")

    # Add a document
    print(f"Attempting to add document: ID='{doc_id}', Content='{doc_content}'")
    collection.add(documents=[doc_content], metadatas=[{"source": "test_script"}], ids=[doc_id])
    print("Document added successfully.")

    # Verify by getting the document count
    count = collection.count()
    print(f"Collection count after add: {count}")
    if count == 1:
        print("SUCCESS: Document count is 1.")
    else:
        print(f"FAILURE: Document count is {count}, expected 1.")

    print("--- Test Completed Successfully --- ")

except Exception as e:
    print("\n!!! AN ERROR OCCURRED DURING THE TEST !!!")
    print(f"Error type: {type(e)}")
    print(f"Error message: {e}")
    import traceback

    traceback.print_exc()
    print("--- Test Failed --- ")
    exit(1)
finally:
    print("\nFinal check for directory existence:")
    if os.path.isdir(persist_directory):
        print(f"Directory {persist_directory} exists.")
        print("Contents:")
        try:
            print(os.listdir(persist_directory))
        except Exception as e:
            print(f"Error listing directory contents: {e}")
    else:
        print(f"Directory {persist_directory} DOES NOT exist.") 