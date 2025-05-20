import chromadb
import sys
from pathlib import Path

print("--- ChromaDB Library Diagnosis ---")
print(f"ChromaDB version: {getattr(chromadb, '__version__', 'N/A')}")
print(f"ChromaDB library loaded from: {getattr(chromadb, '__file__', 'N/A')}")

persistent_client_exists = hasattr(chromadb, 'PersistentClient')
print(f"PersistentClient class exists: {persistent_client_exists}")

if persistent_client_exists:
    print("Attempting to instantiate PersistentClient with a temporary path...")
    temp_db_path = Path("./temp_chroma_diag_db_delete_me")
    try:
        if temp_db_path.exists():
            import shutil
            shutil.rmtree(temp_db_path)
        
        client = chromadb.PersistentClient(path=str(temp_db_path))
        print(f"SUCCESS: chromadb.PersistentClient instantiated successfully at {temp_db_path.resolve()}")
        # Clean up
        if temp_db_path.exists():
            import shutil
            shutil.rmtree(temp_db_path)
            print(f"Cleaned up temporary directory: {temp_db_path.resolve()}")
    except Exception as e:
        print(f"ERROR instantiating PersistentClient: {e}")
        print(f"  Make sure the directory '{temp_db_path.resolve()}' is writable and not locked if it exists.")
else:
    print("PersistentClient class was not found in the loaded chromadb module.")

print("\n--- sys.path (Python's module search paths) ---")
for p in sys.path:
    print(p)

print("\n--- Diagnosis Complete ---") 