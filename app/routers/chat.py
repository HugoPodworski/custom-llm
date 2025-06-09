import os
import json
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse
from app.config import client, groq_client, langfuse
from app.services.cost_service import calculate_cost
from app.services.qdrant_service import search_scenarios_in_qdrant_async
from langfuse.openai import openai
import httpx

router = APIRouter()

def get_recent_messages(messages):
    assistant_contents = [None, None]  # [second_to_last, last]
    user_contents = [None, None]      # [second_to_last, last]
    
    for message in reversed(messages):
        role = message.get('role')
        if role in ('tool', 'function') or message.get('tool_call'):
            continue
        content = message.get('content')
        
        if role == 'assistant':
            if assistant_contents[1] is None:
                assistant_contents[1] = content
            elif assistant_contents[0] is None:
                assistant_contents[0] = content
        elif role == 'user':
            if user_contents[1] is None:
                user_contents[1] = content
            elif user_contents[0] is None:
                user_contents[0] = content
                
        if all(assistant_contents) and all(user_contents):
            break
    
    # Format the output in chronological order
    return f"Assistant: {assistant_contents[0] or 'N/A'}\nUser: {user_contents[0] or 'N/A'}\nAssistant: {assistant_contents[1] or 'N/A'}\nUser: {user_contents[1] or 'N/A'}"

def system_prompt_inject(trieve_response, messages):
    if not messages or messages[0].get('role') != 'system':
        return messages
    messages[0]['content'] += f"\n\nRelevant context+guidelines:\n{trieve_response}"
    return messages

@router.post("/langfuse/trace")
async def langfuse_trace(request: Request):
    payload = await request.json()
    session_id = payload.get('session_id')
    assistant_name = payload.get('assistant_name')
    langfuse.trace(name=session_id, id=session_id, tags=[str(assistant_name)])
    return {"status": "ok"}
    
@router.post("/chat/completions")
async def chat_proxy(request: Request):
    try:
        start_time = time.time()
        payload = await request.json()
        print(f"Raw Payload: {payload}")
        # Extract call_id from payload to use as session_id
        session_id = payload.get('call', {}).get('id')
        
        keys_to_remove = ['call', 'metadata', 'activeAssistant', 'credentials', 'toolDefinitionsExcluded', 'customer', 'phoneNumber', 'assistant', 'timestamp']
        for key in keys_to_remove:
            if key in payload:
                del payload[key]
        
        rag_query = get_recent_messages(payload.get('messages', []))
        
        rag_search_start_time = time.time()
        
        qdrant_results = []
        # Access client and model from app.state via the request object
        current_qdrant_client = request.app.state.qdrant_client
        current_embedding_model = request.app.state.embedding_model

        if current_qdrant_client and current_embedding_model:
            qdrant_results = await search_scenarios_in_qdrant_async(
                qdrant_client_instance=current_qdrant_client,
                embedding_model_instance=current_embedding_model,
                query_text=rag_query,
                top_k=5 
            )
        else:
            print("Qdrant client or embedding model not initialized from app.state. Skipping Qdrant search in /chat/completions.")

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

        payload['stream_options'] = {"include_usage": True}

        payload['trace_id'] = session_id
        payload['name'] = session_id

        print(f"Payload: {payload}")
        
        model_name = payload.get("model", "default")
        
        response_text = []

        if model_name == "llama-3.3-70b-versatile":
            stream = await groq_client.chat.completions.create(**payload)
        else:
            stream = await client.chat.completions.create(**payload)
        
        async def logging_event_stream():
            prompt_tokens = 0
            completion_tokens = 0
            _ttft_logged = False
            last_chunk = None  # Track the last chunk
            nonlocal start_time, response_text

            try:
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        delta_content = chunk.choices[0].delta.content
                        response_text.append(delta_content)
                        
                        if not _ttft_logged:
                            ttft = time.time() - start_time
                            print(f"TTFT: {ttft:.3f} seconds")
                            _ttft_logged = True
                        
                        completion_tokens += len(delta_content) // 4 + 1
                    
                    if hasattr(chunk, 'usage') and chunk.usage:
                        if hasattr(chunk.usage, 'prompt_tokens'):
                            prompt_tokens = chunk.usage.prompt_tokens
                        if hasattr(chunk.usage, 'completion_tokens'):
                            completion_tokens = chunk.usage.completion_tokens
                    
                    last_chunk = chunk  # Save the last chunk

                    json_data = chunk.model_dump_json()
                    yield f"data: {json_data}\n\n"
                
                final_response = "".join(response_text)
                print(f"Response: {final_response}")
                openai.flush_langfuse()
                # After the stream ends, check usage in last_chunk
                if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage is not None:
                    prompt_tokens = getattr(last_chunk.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(last_chunk.usage, "completion_tokens", 0)
                    print(f"Prompt Tokens: {prompt_tokens}")
                    print(f"Completion Tokens: {completion_tokens}")
                    print(f"Total Tokens: {getattr(last_chunk.usage, 'total_tokens', 0)}")
                else:
                    # Fallback: estimate tokens if usage is missing
                    prompt_text = ""
                    for msg in payload.get('messages', []):
                        if isinstance(msg, dict) and 'content' in msg and msg.get('content'):
                            prompt_text += msg.get('content', '')
                    prompt_tokens = len(prompt_text) // 4 + 1
                
                cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
                
                print(f"Cost: ${cost:.6f}")
                
                total_request_time = time.time() - start_time
                print(f"Total Request Processing Time: {total_request_time:.3f} seconds")
                
            except Exception as ex_stream:
                print(f"Error during stream processing: {ex_stream}")
                final_response = "".join(response_text)
                print(f"Partial Response: {final_response}")
                
                # After the stream ends, check usage in last_chunk
                if last_chunk and hasattr(last_chunk, "usage") and last_chunk.usage is not None:
                    prompt_tokens = getattr(last_chunk.usage, "prompt_tokens", 0)
                    completion_tokens = getattr(last_chunk.usage, "completion_tokens", 0)
                    print(f"Prompt Tokens: {prompt_tokens}")
                    print(f"Completion Tokens: {completion_tokens}")
                    print(f"Total Tokens: {getattr(last_chunk.usage, 'total_tokens', 0)}")
                else:
                    # Fallback: estimate tokens if usage is missing
                    prompt_text = ""
                    for msg in payload.get('messages', []):
                        if isinstance(msg, dict) and 'content' in msg and msg.get('content'):
                            prompt_text += msg.get('content', '')
                    prompt_tokens = len(prompt_text) // 4 + 1
                
                cost = calculate_cost(model_name, prompt_tokens, completion_tokens)
                
                print(f"Cost: ${cost:.6f}")
                
                total_request_time = time.time() - start_time
                print(f"Total Request Processing Time: {total_request_time:.3f} seconds")

        return StreamingResponse(logging_event_stream(), media_type="text/event-stream")
    except Exception as e:
        import traceback
        print(f"Error in /chat/completions endpoint: {str(e)}\nTraceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}") 