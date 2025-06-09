import os
import json
import time
from typing import List
from fastapi import APIRouter, Request, HTTPException
from app.schemas import SearchRequest, ScenarioHit
from app.services.qdrant_service import search_scenarios_in_qdrant_async
import httpx

router = APIRouter()

async def search_trieve(query):
    api_key = os.getenv("TRIEVE_API_KEY")
    dataset_id = os.getenv("TRIEVE_DATASET_ID")
    if not api_key or not dataset_id:
        raise RuntimeError("Missing Trieve API credentials or dataset ID")
    async with httpx.AsyncClient() as client:
        url = "https://api.trieve.ai/api/chunk/search"
        
        headers = {
        "Authorization": api_key,
        "TR-Dataset": dataset_id,
        "Content-Type": "application/json",
        "X-API-Version": "V2",
    }
        
        data = {
            "query": query,
            "search_type": "bm25",
            "content_only": False,
            "get_total_pages": True,
            "page_size": 5,
            "slim_chunks": False,
            "sort_options": {}
        }
        
        response = await client.post(url, headers=headers, json=data)        
        
        lines = []
        for chunk in response.json().get('chunks', []):
            scenario = json.loads(chunk['chunk']['chunk_html'])
            context = scenario.get('context', '').strip()
            guidelines = scenario.get('responseGuidelines', '').strip()
            if context or guidelines:
                lines.append(f"- {context} {guidelines}".strip())
        
        if not lines:
            return "No relevant results found."
        
        return "\n".join(lines)

@router.post("/search", response_model=List[ScenarioHit])
async def search_qdrant_endpoint(request: Request, search_request: SearchRequest): 
    current_qdrant_client = request.app.state.qdrant_client
    current_embedding_model = request.app.state.embedding_model

    if not current_qdrant_client or not current_embedding_model:
        print("Search request received, but Qdrant client or embedding model is not available from app.state.")
        raise HTTPException(status_code=503, detail="Qdrant client or embedding model not initialized. Search is unavailable.")

    print(f"Received Qdrant search request: query='{search_request.query_text}', top_k={search_request.top_k}")
    
    start_time = time.time()
    try:
        results = await search_scenarios_in_qdrant_async(
            qdrant_client_instance=current_qdrant_client,
            embedding_model_instance=current_embedding_model,
            query_text=search_request.query_text, 
            top_k=search_request.top_k
        )
    except Exception as e:
        print(f"Error during search_scenarios_in_qdrant_async execution in /search endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during Qdrant search: {str(e)}")
    
    end_time = time.time()
    print(f"Total Qdrant /search request processing time: {end_time - start_time:.4f} seconds")

    if not results:
        print("No results found from Qdrant or search failed in /search endpoint.")
    
    print(f"Returning {len(results)} scenarios from Qdrant in /search endpoint.")
    return results

@router.post("/trieve-search")
async def trieve_search(request: Request): 
    try:
        start_time = time.time()
        
        url = "https://api.trieve.ai/api/health"
        print(f"Performing Trieve health check on URL: {url} using httpx")

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        
        response.raise_for_status()
        
        response_text = response.text
        print(f"Trieve health check response: {response_text}")
        
        end_time = time.time()
        print(f"Trieve health check processed in {end_time - start_time:.4f} seconds.")
        
        return {"trieve_health_status": "success", "response": response_text}

    except httpx.HTTPStatusError as http_err:
        print(f"Trieve health check HTTP error: {http_err} - Status: {http_err.response.status_code} - Response: {http_err.response.text}")
        raise HTTPException(status_code=http_err.response.status_code, 
                            detail=f"Trieve API health check failed: {str(http_err)} - {http_err.response.text}")
    except httpx.RequestError as req_err:
        print(f"Trieve health check request error: {req_err}")
        raise HTTPException(status_code=503,
                            detail=f"Trieve API health check request failed: {str(req_err)}")
    except Exception as e:
        print(f"Unexpected error in Trieve health check endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during Trieve health check: {str(e)}") 