"""ChromaDB Query Operations"""

async def chroma_query_documents(query_text: str, collection_name: str = "default", **kwargs):
    """Query documents in ChromaDB collection"""
    try:
        from chungoid.utils.chroma_utils import get_chroma_client
        
        client = get_chroma_client()
        collection = client.get_collection(collection_name)
        
        results = collection.query(
            query_texts=[query_text],
            n_results=kwargs.get('n_results', 10)
        )
        
        return {
            "success": True,
            "results": results,
            "collection": collection_name
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

# Compatibility alias for agents calling chromadb_query_documents
async def chromadb_query_documents(query: str, collection: str = None, **kwargs):
    """Compatibility alias for chroma_query_documents"""
    if collection:
        kwargs['collection_name'] = collection
    return await chroma_query_documents(query_text=query, **kwargs)
