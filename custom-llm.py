import os
import json
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from starlette.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx

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
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

async def search_trieve(query: str) -> str:
    """Call Trieve.ai chunk search API and return formatted context+guidelines."""
    api_key = os.getenv("TRIEVE_API_KEY")
    dataset_id = os.getenv("TRIEVE_DATASET_ID")
    if not api_key or not dataset_id:
        raise RuntimeError("Missing Trieve API credentials or dataset ID")

    url = "https://api.trieve.ai/api/chunk/search"
    headers = {
        "Authorization": api_key,
        "TR-Dataset": dataset_id,
        "Content-Type": "application/json",
        "X-API-Version": "V2",
    }
    payload = {"query": query, "search_type": "bm25", "page_size": 5}

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()

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
        stream_time = time.time()
        stream = await client.chat.completions.create(**payload)
        
        ttft = []
        async def event_stream():
            first_token = True
            async for chunk in stream:
                # Log TTFT on first chunk
                if first_token:
                    end_time = time.time()
                    ttft = end_time - stream_time
                    ttft.append(ttft)
                    print(f"TTFT: {ttft:.3f} seconds")
                    first_token = False
                # Serialize the full chunk as JSON
                json_data = chunk.model_dump_json()
                yield f"data: {json_data}\n\n"

        total_time = time.time() - start_time
        print(f"OPERATIONS: {total_time} - {ttft} - {trieve_speed}")
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")