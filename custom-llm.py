import os
import json
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Qdrant and Embedding Model Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "liv-scenarios"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"

print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = TextEmbedding(model_name=EMBEDDING_MODEL_NAME, cache_dir=".")
print("Embedding model initialized.")

qdrant_client = None
if QDRANT_URL:
    try:
        print(f"Connecting to Qdrant at {QDRANT_URL}...")
        qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
        print("Qdrant client initialized successfully.")
        # Optional: Test connection (can be un-commented if needed)
        # collection_info = qdrant_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)
        # print(f"Successfully connected to collection '{QDRANT_COLLECTION_NAME}'. Status: {collection_info.status}")
    except Exception as e:
        print(f"Error initializing Qdrant client: {e}")
        print("Please ensure Qdrant is running and accessible, and credentials are correct.")
else:
    print("QDRANT_URL environment variable not set. Qdrant client will not be initialized. Qdrant search functionality will be disabled.")

# Pydantic models for /search endpoint (similar to ragpipeline.py)
class SearchRequest(BaseModel):
    query_text: str
    top_k: Optional[int] = 5

class ScenarioHit(BaseModel):
    id: Any
    score: float
    payload: Dict[str, Any]

# Helper functions for Qdrant search (adapted from ragpipeline.py)
def embed_query_text(query_text: str) -> Optional[List[float]]:
    try:
        # Assuming embedding_model is initialized globally as in ragpipeline.py
        embedding = list(embedding_model.embed([query_text]))[0]
        return embedding.tolist()
    except Exception as e:
        print(f"Error embedding query '{query_text}': {e}")
        return None

def search_scenarios_in_qdrant_sync(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    global qdrant_client # Ensure we're using the global client
    if not qdrant_client:
        print("Qdrant client not initialized. Skipping Qdrant search.")
        return []

    t_embedding_start = time.time()
    print(f"Embedding query: '{query_text}'...")
    query_embedding = embed_query_text(query_text)
    t_embedding_end = time.time()
    print(f"  Embedding time: {t_embedding_end - t_embedding_start:.4f} seconds")

    if not query_embedding:
        print("Failed to generate query embedding. Skipping search.")
        return []
    
    t_qdrant_call_start = time.time()
    print(f"Searching Qdrant collection '{QDRANT_COLLECTION_NAME}' with embedded query...")
    try:
        search_result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        actual_hits = []
        # Simplified hit extraction, assuming search_result is a list of ScoredPoint-like objects
        # or an object with a .points attribute that is a list of ScoredPoint-like objects.
        # This part might need adjustment based on the exact version/behavior of qdrant_client
        if hasattr(search_result, 'points'):
             actual_hits = search_result.points
        elif isinstance(search_result, list): # If search_result is directly a list of hits
            actual_hits = search_result
        elif isinstance(search_result, tuple) and len(search_result) > 0 and hasattr(search_result[0], 'points'): # As in original ragpipeline
            actual_hits = search_result[0].points
        elif isinstance(search_result, tuple) and len(search_result) > 0 and isinstance(search_result[0], list): # As in original ragpipeline
            actual_hits = search_result[0]
        else:
            print(f"Warning: Unexpected structure from query_points response: {type(search_result)}")
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
        print(f"Error searching Qdrant: {e}")
        return []
    finally:
        t_qdrant_call_end = time.time()
        # Ensure t_qdrant_call_start was defined
        if 't_qdrant_call_start' in locals() or 't_qdrant_call_start' in globals():
            print(f"  Qdrant call + result processing time: {t_qdrant_call_end - t_qdrant_call_start:.4f} seconds")

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

def get_recent_messages(messages):
    # Get the content of the most recent user and assistant messages, ignoring any tool/function calls
    assistant_content = None
    user_content = None
    for message in reversed(messages):
        role = message.get('role')
        # skip tool or function call messages
        if role in ('tool', 'function') or message.get('tool_call'):
            continue
        content = message.get('content')
        if role == 'assistant' and assistant_content is None:
            assistant_content = content
        if role == 'user' and user_content is None:
            user_content = content
        if assistant_content is not None and user_content is not None:
            break
    return f"User: {user_content}\nAssistant: {assistant_content}"

def system_prompt_inject(trieve_response, messages):
    """Append Knowledge Base Results into the system prompt of the messages and return the updated list."""
    # Only modify if the first message is the system prompt
    if not messages or messages[0].get('role') != 'system':
        return messages
    # Append Knowledge Base Results to the system prompt
    messages[0]['content'] += f"\n\nRelevant context+guidelines:\n{trieve_response}"
    return messages

@app.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        start_time = time.time()
        payload = await request.json()
        print(payload)

        keys_to_remove = ['call', 'metadata', 'activeAssistant', 'credentials', 'toolDefinitionsExcluded', 'customer', 'phoneNumber', 'assistant', 'timestamp']
        for key in keys_to_remove:
            if key in payload:
                del payload[key]
        
        trieve_query = get_recent_messages(payload['messages'])

        trieve_time = time.time()
        trieve_response = await search_trieve(trieve_query)
        trieve_speed = time.time() - trieve_time
        print(f"TRIEVE: {trieve_speed:.3f} seconds")
        # Inject the Knowledge Base Results via helper
        payload['messages'] = system_prompt_inject(trieve_response, payload.get('messages', [])) or payload.get('messages', [])

        # Create the streaming completion
        stream_start = time.time()
        stream = await client.chat.completions.create(**payload)
        
        # Use a list to capture the ttft value
        captured_ttft = [None]
        
        async def event_stream():
            first_token = True
            async for chunk in stream:
                # Log TTFT on first chunk
                if first_token:
                    ttft = time.time() - stream_start
                    captured_ttft[0] = ttft
                    print(f"TTFT: {ttft:.3f} seconds")
                    total_time = time.time() - start_time
                    # operations_time = total_time - ttft - trieve_speed
                    # print(f"OPERATIONS: {operations_time:.8f} seconds")
                    first_token = False
                # Serialize the full chunk as JSON
                json_data = chunk.model_dump_json()
                yield f"data: {json_data}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/search", response_model=List[ScenarioHit])
async def search_qdrant_endpoint(search_request: SearchRequest): # Changed request to search_request to match Pydantic model
    global qdrant_client # Ensure we're using the global client
    if not qdrant_client:
        print("Search request received, but Qdrant client is not available.")
        raise HTTPException(status_code=503, detail="Qdrant client not initialized. Search is unavailable.")

    print(f"Received Qdrant search request: query='{search_request.query_text}', top_k={search_request.top_k}")
    
    start_time = time.time()
    try:
        # Use run_in_threadpool for the synchronous Qdrant search function
        results = await run_in_threadpool(
            search_scenarios_in_qdrant_sync, 
            query_text=search_request.query_text, 
            top_k=search_request.top_k
        )
    except Exception as e:
        print(f"Error during search_scenarios_in_qdrant_sync execution in threadpool: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during Qdrant search: {str(e)}")
    
    end_time = time.time()
    print(f"Total Qdrant request processing time: {end_time - start_time:.4f} seconds")

    if not results:
        print("No results found from Qdrant or search failed.")
        # Returning an empty list is consistent with ragpipeline.py
    
    print(f"Returning {len(results)} scenarios from Qdrant.")
    return results

# Optional: Add a root endpoint if desired, similar to ragpipeline
# @app.get("/")
# async def root():
#     return {"message": "Custom LLM API with Trieve and Qdrant search is running."}

# Note: The __main__ block for uvicorn.run is not included here as custom-llm.py
# might be run differently (e.g. via an external ASGI server like uvicorn directly).
# If you need it, it can be added similarly to ragpipeline.py.