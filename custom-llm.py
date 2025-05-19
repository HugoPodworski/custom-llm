import os
import json
import time

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from openai import AsyncOpenAI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
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

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Qdrant and Embedding Model Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = "liv-scenarios-2"
EMBEDDING_MODEL_NAME = "static-retrieval-mrl-en-v1"

print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
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

MODEL_PRICES = {
    "default": {"prompt": 0.0005 / 1000, "completion": 0.0015 / 1000},
    "gpt-4.1-mini": {"prompt": 0.0004 / 1000, "completion": 0.0016 / 1000},
    # Add more models and their pricing here
}

def calculate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates the cost of an LLM request based on model, prompt tokens, and completion tokens."""
    pricing_key = "default"  # Fallback to default pricing

    # Find the most specific matching key
    best_match_len = 0
    for key in MODEL_PRICES:
        if model_name.startswith(key) and key != "default":
            if len(key) > best_match_len:
                pricing_key = key
                best_match_len = len(key)
    
    if best_match_len == 0 and model_name in MODEL_PRICES: # Exact match for a non-default key
        pricing_key = model_name

    prices = MODEL_PRICES.get(pricing_key)

    if not prices: # Should ideally not happen if "default" is correctly set up
        print(f"Warning: Pricing not found for model key '{pricing_key}' (original model: '{model_name}'). Using zero rates.")
        return 0.0
    
    prompt_cost = prompt_tokens * prices["prompt"]
    completion_cost = completion_tokens * prices["completion"]
    total_cost = prompt_cost + completion_cost
    # print(f"Cost calc: model='{model_name}', matched_key='{pricing_key}', p_tokens={prompt_tokens}, c_tokens={completion_tokens}, cost={total_cost}")
    return total_cost

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
        # Assuming embedding_model is initialized globally
        embedding = embedding_model.encode([query_text])[0]
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
    query_embedding = embed_query_text(query_text)
    t_embedding_end = time.time()
    print(f"Embedding Time: {t_embedding_end - t_embedding_start:.4f} seconds")

    if not query_embedding:
        print("Failed to generate query embedding. Skipping search.")
        return []
    
    t_qdrant_call_start = time.time()
    try:
        search_result = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_embedding,
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
            print(f"Qdrant Search Time: {t_qdrant_call_end - t_qdrant_call_start:.4f} seconds")

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

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the service is running.
    """
    return {"status": "ok"}

@app.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        start_time = time.time()
        payload = await request.json()
        keys_to_remove = ['call', 'metadata', 'activeAssistant', 'credentials', 'toolDefinitionsExcluded', 'customer', 'phoneNumber', 'assistant', 'timestamp']
        for key in keys_to_remove:
            if key in payload:
                del payload[key]
        
        rag_query = get_recent_messages(payload.get('messages', []))
        # Remove verbose logging
        # print(f"Generated RAG query: {rag_query}")

        rag_search_start_time = time.time()
        
        qdrant_results = []
        if qdrant_client:
            # Remove verbose logging
            # print(f"Qdrant client found, proceeding with search for query: '{rag_query}'")
            qdrant_results = await run_in_threadpool(
                search_scenarios_in_qdrant_sync,
                query_text=rag_query,
                top_k=5 
            )
        else:
            print("Qdrant client not initialized. Skipping Qdrant search.")

        formatted_rag_results = []
        if qdrant_results:
            for hit in qdrant_results:
                context = hit.get("payload", {}).get("context", "").strip()
                guidelines = hit.get("payload", {}).get("responseGuidelines", "").strip()
                if context or guidelines:
                    formatted_rag_results.append(f"- {context} {guidelines}".strip())
        
        rag_response_string = "\n".join(formatted_rag_results)
        if not rag_response_string: 
            rag_response_string = "No relevant context found from knowledge base."

        rag_search_speed = time.time() - rag_search_start_time
        print(f"Total RAG Time: {rag_search_speed:.3f} seconds")
        
        payload['messages'] = system_prompt_inject(rag_response_string, payload.get('messages', []))

        # Print payload - keep for debugging but show shortened version
        truncated_payload = {**payload}
        if 'messages' in truncated_payload:
            messages_preview = []
            for msg in truncated_payload['messages']:
                content = msg.get('content', '')
                content_preview = content[:100] + '...' if len(content) > 100 else content
                messages_preview.append({**msg, 'content': content_preview})
            truncated_payload['messages'] = messages_preview
        print(f"Payload: {truncated_payload}")
        
        # Track the model being used for cost calculation
        model_name = payload.get("model", "default")
        
        # Create a response tracking list for accumulating full response
        response_text = []

        stream = await client.chat.completions.create(**payload)
        
        async def logging_event_stream():
            # Values for accumulating response and usage data
            prompt_tokens = 0
            completion_tokens = 0
            _ttft_logged = False
            # TTFT is calculated relative to start_time of the request
            nonlocal start_time, response_text

            try:
                async for chunk in stream:
                    # Process content for response accumulation
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        delta_content = chunk.choices[0].delta.content
                        response_text.append(delta_content)
                        
                        # Log TTFT on first content chunk
                        if not _ttft_logged:
                            ttft = time.time() - start_time
                            print(f"TTFT: {ttft:.3f} seconds")
                            _ttft_logged = True
                        
                        # Count completion tokens (approximate by characters/4)
                        completion_tokens += len(delta_content) // 4 + 1
                    
                    # If usage data is available, use it instead of our approximation
                    if hasattr(chunk, 'usage') and chunk.usage:
                        if hasattr(chunk.usage, 'prompt_tokens'):
                            prompt_tokens = chunk.usage.prompt_tokens
                        if hasattr(chunk.usage, 'completion_tokens'):
                            completion_tokens = chunk.usage.completion_tokens
                    
                    # Stream the chunk to client
                    json_data = chunk.model_dump_json()
                    yield f"data: {json_data}\n\n"
                
                # At end of stream, compute cost based on our token count or API-provided count
                final_response = "".join(response_text)
                print(f"Response: {final_response}")
                
                # If prompt_tokens is still 0, estimate from payload
                if prompt_tokens == 0:
                    # Rough estimate: count all characters in messages and divide by 4
                    prompt_text = ""
                    for msg in payload.get('messages', []):
                        if isinstance(msg, dict) and 'content' in msg and msg.get('content'):
                            prompt_text += msg.get('content', '')
                    prompt_tokens = len(prompt_text) // 4 + 1
                
                cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
                
                print(f"Prompt Tokens: {prompt_tokens}")
                print(f"Completion Tokens: {completion_tokens}")
                print(f"Cost: ${cost:.6f}")
                
                total_request_time = time.time() - start_time
                print(f"Total Request Processing Time: {total_request_time:.3f} seconds")
                
            except Exception as ex_stream:
                print(f"Error during stream processing: {ex_stream}")
                # Calculate what we have so far
                final_response = "".join(response_text)
                print(f"Partial Response: {final_response}")
                
                # If prompt_tokens is still 0, estimate from payload
                if prompt_tokens == 0:
                    # Rough estimate: count all characters in messages and divide by 4
                    prompt_text = ""
                    for msg in payload.get('messages', []):
                        if isinstance(msg, dict) and 'content' in msg and msg.get('content'):
                            prompt_text += msg.get('content', '')
                    prompt_tokens = len(prompt_text) // 4 + 1
                
                cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
                
                print(f"Prompt Tokens: {prompt_tokens}")
                print(f"Completion Tokens: {completion_tokens}")
                print(f"Cost: ${cost:.6f}")
                
                total_request_time = time.time() - start_time
                print(f"Total Request Processing Time: {total_request_time:.3f} seconds")

        return StreamingResponse(logging_event_stream(), media_type="text/event-stream")
    except Exception as e:
        # Log the full traceback for better debugging
        import traceback
        print(f"Error in /chat/completions endpoint: {str(e)}\nTraceback: {traceback.format_exc()}")
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

@app.post("/trieve-search")
async def trieve_search(request: Request): # request parameter is unused for this health check
    """
    Perform a health check on the Trieve API using httpx.
    Note: This endpoint currently performs a health check.
    """
    try:
        start_time = time.time()
        
        url = "https://api.trieve.ai/api/health"
        print(f"Performing Trieve health check on URL: {url} using httpx")

        async with httpx.AsyncClient() as client:
            response = await client.get(url)
        
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        response_text = response.text
        print(f"Trieve health check response: {response_text}")
        
        end_time = time.time()
        print(f"Trieve health check processed in {end_time - start_time:.4f} seconds.")
        
        return {"trieve_health_status": "success", "response": response_text}

    except httpx.HTTPStatusError as http_err:
        # For httpx.HTTPStatusError, the response is http_err.response
        print(f"Trieve health check HTTP error: {http_err} - Status: {http_err.response.status_code} - Response: {http_err.response.text}")
        raise HTTPException(status_code=http_err.response.status_code, 
                            detail=f"Trieve API health check failed: {str(http_err)} - {http_err.response.text}")
    except httpx.RequestError as req_err:
        # For other httpx request errors (e.g., network issues)
        print(f"Trieve health check request error: {req_err}")
        raise HTTPException(status_code=503, # Service Unavailable
                            detail=f"Trieve API health check request failed: {str(req_err)}")
    except Exception as e:
        print(f"Unexpected error in Trieve health check endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during Trieve health check: {str(e)}")