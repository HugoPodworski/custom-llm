import os
import json
import time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
from starlette.middleware.cors import CORSMiddleware

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

@app.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        start_time = time.time()
        payload = await request.json()
        payload["stream"] = True
        
        # Create the streaming completion
        stream = await client.chat.completions.create(**payload)
        
        async def event_stream():
            first_token = True
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token:
                        end_time = time.time()
                        ttft = end_time - start_time
                        print(f"TTFT: {ttft:.3f} seconds")
                        first_token = False
                    # Emit each piece as its own SSE
                    yield f"data: {json.dumps({'content': delta})}\n\n"
            # Signal done
            yield "data: [DONE]\n\n"
            
        return StreamingResponse(event_stream(), media_type="text/event-stream")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")