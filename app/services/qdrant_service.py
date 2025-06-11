import time
from typing import List, Optional, Dict, Any
from fastapi.concurrency import run_in_threadpool
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
from app.config import QDRANT_COLLECTION_NAME

# Helper function for embedding text
def embed_query_text(query_text: str, embedding_model_instance: SentenceTransformer) -> Optional[List[float]]:
    try:
        embedding = embedding_model_instance.encode([query_text])[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error embedding query '{query_text}': {e}")
        return None

# Asynchronous Qdrant search function
async def search_scenarios_in_qdrant_async(
    qdrant_client_instance: AsyncQdrantClient, 
    embedding_model_instance: SentenceTransformer,
    query_text: str, 
    top_k: int = 5
) -> List[Dict[str, Any]]:
    if not qdrant_client_instance:
        print("AsyncQdrantClient not available (from app.state). Skipping Qdrant search.")
        return []
    if not embedding_model_instance:
        print("Embedding model not available (from app.state). Skipping Qdrant search.")
        return []

    t_embedding_start = time.time()
    query_embedding = await run_in_threadpool(embed_query_text, query_text, embedding_model_instance)
    t_embedding_end = time.time()
    print(f"Embedding Time: {t_embedding_end - t_embedding_start:.4f} seconds")

    if not query_embedding:
        print("Failed to generate query embedding. Skipping search.")
        return []
    
    t_qdrant_call_start = time.time()
    try:
        search_result = await qdrant_client_instance.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        actual_hits = []
        if isinstance(search_result, list): # Async client's query_points directly returns a list of ScoredPoint
            actual_hits = search_result
        elif hasattr(search_result, 'points'): # Fallback, though async client usually returns list
             actual_hits = search_result.points
        else:
            print(f"Warning: Unexpected structure from async query_points response: {type(search_result)}")
            actual_hits = []

        results_as_dicts = []
        for hit in actual_hits: 
            results_as_dicts.append({
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload
            })
        return results_as_dicts
    except Exception as e:
        print(f"Error searching Qdrant with async client: {e}")
        return []
    finally:
        t_qdrant_call_end = time.time()
        # Ensure t_qdrant_call_start was defined (it should be if try block was entered)
        print(f"Qdrant Async Search Time: {t_qdrant_call_end - t_qdrant_call_start:.4f} seconds") 